#!/usr/bin/env python3
"""Report torch.export compatibility, weight size, and FLOPs for every timm model.

Builds each registered timm architecture with random (non-pretrained) weights on the
``meta`` device, attempts ``torch.export.export()``, and records success/failure, weight
size, and inference FLOPs. Results are written to a Markdown file (default ``models.md``
at the repo root) that is both the human-readable report and the format this script
re-parses for --resume.

Each model is evaluated in an isolated subprocess (crashes/hangs/OOM in one model must not
take down the whole run), with cache directories redirected into a per-worker temp dir that
is deleted on exit so no artifacts accumulate across a run of ~1300 models.
"""
import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import sys
import time

import yaml

from exportlib import MAX_RES, cpu_count, globs, pretrained_info, resolved_input_size, run_worker as _run_worker


def run_worker(model_name, max_res, timeout, dynamic_timeout, ops_timeout, collect, excluded=False):
    argv = ['--worker', model_name, '--max-res', max_res,
            '--dynamic-timeout', dynamic_timeout, '--ops-timeout', ops_timeout]
    if not collect:
        argv.append('--no-ops')
    if excluded:
        argv.append('--excluded')
    return _run_worker(__file__, argv, model_name, timeout)


# Structural fact from how every example input in this script is built -- always
# (batch, channel, height, width) -- not a guess about what a given model "means" by its dims.
_AXIS_NAMES = {0: 'B', 1: 'C', 2: 'H', 3: 'W'}


def describe_dynamic_shapes(ep):
    """Render every free symbol torch.export retained in `ep`, verbatim from the graph's own
    metadata: no invented labels, just each dynamic dim's axis (substituted in place of the
    graph's internal symbol name, e.g. 's53' -> 'H', via the structural dim->axis mapping
    above) and its real lower bound from ep.range_constraints. Dims that collapsed onto the
    same symbol (e.g. an architecture that only accepts square input) naturally report once,
    combined, since they *are* the same symbol -- no special-casing needed. Any symbol not
    traceable to one of the four input dims (unrelated data-dependent shape elsewhere in the
    model) is still reported, by its raw graph symbol name, rather than silently dropped.
    """
    input_names = set(ep.graph_signature.user_inputs)
    sym_to_axes = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != 'placeholder' or node.name not in input_names:
            continue
        val = node.meta.get('val')
        if not hasattr(val, 'shape'):
            continue
        for dim, size in enumerate(val.shape):
            expr = getattr(size, 'node', None) and size.node.expr
            if expr is None or not expr.free_symbols:
                continue  # concrete dim, e.g. batch=1 -- not a torch.export.Dim, no symbol
            for sym in expr.free_symbols:
                sym_to_axes.setdefault(sym, []).append(_AXIS_NAMES.get(dim, f'dim{dim}'))

    if not ep.range_constraints:
        return None

    parts = []
    for sym in sorted(ep.range_constraints, key=str):
        rc = ep.range_constraints[sym]
        axes = sym_to_axes.get(sym)
        label = '='.join(axes) if axes else str(sym)
        # ValueRanges is always a genuine (lower, upper) pair -- represent both consistently
        # as an interval rather than special-casing the (common) unbounded-upper case.
        upper = '∞' if str(rc.upper) == 'int_oo' else str(rc.upper)
        parts.append(f'{label}∈[{rc.lower},{upper}]')
    return ' '.join(parts) if parts else None


# torch.export retains SymInt arguments for two structurally different things: tensor
# extents (a `view` size, a `slice` bound), whose values are a function of the input
# resolution, and genuine architectural knobs that merely happen to be SymInt-typed. Only
# the latter belong in an operator's configuration -- recording the former verbatim
# explodes the catalog (measured: `view.size` alone contributes 533 distinct values across
# 30 models, vs 6 once abstracted), so SymInt args are recorded as arity except for the
# ops listed here. An op not listed defaults to abstraction, which is the safe direction:
# a new op can never blow the catalog up, only under-describe itself.
SYMINT_LITERAL_OPS = {
    'aten.convolution.default',
    'aten.constant_pad_nd.default',
}

# `device` differs between the meta and CPU export paths this script already takes per
# model, so recording it would make a model's configuration depend on which path it
# happened to take rather than on the architecture. `layout`/`pin_memory` are invariant
# noise.
DROPPED_ARGS = {'device', 'layout', 'pin_memory'}

# An export bookkeeping node, not computation a backend has to implement.
DROPPED_OPS = {'aten._assert_tensor_metadata.default'}

_DTYPE_NAMES = {
    'torch.float32': 'f32', 'torch.float64': 'f64', 'torch.float16': 'f16',
    'torch.bfloat16': 'bf16', 'torch.int64': 'i64', 'torch.int32': 'i32',
    'torch.int16': 'i16', 'torch.int8': 'i8', 'torch.uint8': 'u8', 'torch.bool': 'bool',
}

_SYMINT_ARGS_CACHE = {}

# One `name: type` pair of a schema's argument list, e.g. `SymInt[] stride` -- the split
# below has already isolated it, so the type is everything up to the last whitespace.
_SCHEMA_ARG_RE = re.compile(r'^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z_0-9]*)$')


def symint_arg_names(schema):
    """Names of `schema`'s SymInt-typed arguments, read out of the schema *text*.

    The JIT type system erases SymInt -- `str(arg.type)` reports `List[int]` for both
    `int[] dims` (permutations: a real configuration) and `SymInt[] size` (a tensor
    extent), so the distinction only survives in the declaration string itself, e.g.
    `aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)`.
    """
    text = str(schema)
    cached = _SYMINT_ARGS_CACHE.get(text)
    if cached is not None:
        return cached

    body = text[text.index('(') + 1:text.rindex(') ->')]
    names = set()
    # Split on commas that are not inside a bracketed type/default, e.g. `int[2] stride=[]`.
    for part in re.split(r',(?![^\[]*\])', body):
        part = part.strip().lstrip('*').strip()
        part = part.split('=', 1)[0].strip()  # drop the default value
        m = _SCHEMA_ARG_RE.match(part)
        if m and 'SymInt' in m.group('type'):
            names.add(m.group('name'))
    _SYMINT_ARGS_CACHE[text] = names
    return names


def _plain(value):
    """Coerce a schema argument value to something JSON/YAML can round-trip."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)  # torch.contiguous_format, torch.float32, ...


def _meta_val(arg):
    """The fake tensor an argument carries, for arguments that are graph nodes at all
    (a node's args are just as often plain ints or lists)."""
    meta = getattr(arg, 'meta', None)
    return meta.get('val') if meta else None


def _concrete_shape(val):
    """`val`'s shape as plain ints, or None if it is absent or symbolic."""
    if val is None or not hasattr(val, 'shape'):
        return None
    try:
        return [int(d) for d in val.shape]
    except Exception:
        return None  # data-dependent/dynamic dim -- no honest concrete answer


def op_config(node):
    """(op name, configuration) for one call_function node of an exported graph.

    The configuration is the node's non-Tensor schema arguments (the knobs a backend has
    to honour: stride, eps, dim, keepdim, ...) with tensor-extent SymInts abstracted per
    SYMINT_LITERAL_OPS, plus a few facts derived from the graph's own shape metadata that
    the argument list does not carry -- most importantly a convolution's kernel size and
    whether it is depthwise, which is the difference between two very different kernels
    wearing the same `aten.convolution.default` name.
    """
    target = str(node.target)
    schema = node.target._schema
    symints = symint_arg_names(schema)
    abstract_symints = target not in SYMINT_LITERAL_OPS

    config = {}
    for i, arg in enumerate(schema.arguments):
        if 'Tensor' in str(arg.type) or arg.name in DROPPED_ARGS:
            continue
        value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name, arg.default_value)
        if abstract_symints and arg.name in symints:
            value = f'[*{len(value)}]' if isinstance(value, (list, tuple)) else '*'
        config[arg.name] = _plain(value)

    # Derived facts are prefixed `out_` rather than named `dtype`/`rank`: several schemas
    # (mean.dim, arange, to.dtype) already carry a `dtype` argument of their own, and the
    # requested dtype and the produced one are different facts.
    out = node.meta.get('val')
    if isinstance(out, (list, tuple)) and out:
        out = out[0]  # multi-output op (batch_norm, layer_norm) -- describe its primary result
    dtype = getattr(out, 'dtype', None)
    if dtype is not None:
        config['out_dtype'] = _DTYPE_NAMES.get(str(dtype), str(dtype))
    out_shape = _concrete_shape(out)
    if out_shape is not None:
        config['out_rank'] = len(out_shape)

    if 'convolution' in target or target.startswith('aten.conv'):
        weight = _concrete_shape(_meta_val(node.args[1])) if len(node.args) > 1 else None
        if weight and len(weight) > 2:
            config['kernel'] = weight[2:]
        inputs = _concrete_shape(_meta_val(node.args[0])) if node.args else None
        groups = config.get('groups')
        if inputs and len(inputs) > 1 and isinstance(groups, int):
            config['depthwise'] = groups > 1 and groups == inputs[1]

    return target, config


def canonical_config(config):
    """Order-independent identity of a configuration, for deduplication and id assignment.

    Configurations themselves keep their schema argument order (`stride` before `padding`
    before `dilation`), which is how a human reads them; this is only the key.
    """
    return json.dumps(config, sort_keys=True)


def collect_ops(ep):
    """([[op name, configuration, node count], ...], {op name: schema}) for an exported program.

    call_function nodes without a schema (`operator.getitem`, higher-order ops) are
    structural graph plumbing rather than operators a backend lowers, so they are skipped.
    """
    counts = {}
    configs = {}  # keyed the same, but keeping the schema-ordered dict for output
    schemas = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != 'call_function' or not hasattr(node.target, '_schema'):
            continue
        target, config = op_config(node)
        if target in DROPPED_OPS:
            continue
        schemas[target] = str(node.target._schema).replace('aten::', '', 1)
        key = (target, canonical_config(config))
        counts[key] = counts.get(key, 0) + 1
        configs.setdefault(key, config)
    ops = [[target, configs[(target, config)], count] for (target, config), count in sorted(counts.items())]
    return ops, schemas


@contextlib.contextmanager
def time_budget(seconds):
    """Bound a block on its own SIGALRM budget, raising TimeoutError when it overruns.

    Both post-export sub-checks (dynamic shapes, op collection) can run far longer than
    the export itself on a minority of architectures, and neither is worth losing the
    already-computed results over -- so each gets a budget independent of --timeout.
    """
    import signal

    def _alarm(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def worker_main(model_name, max_res, dynamic_timeout, ops_timeout, collect, excluded=False):
    import torch
    import torch.utils.flop_counter as flop_counter
    import timm
    from timm.models import model_entrypoint

    result = {'name': model_name, 'status': None, 'error': None}
    try:
        result['family'] = model_entrypoint(model_name).__module__.rsplit('.', 1)[-1]
    except Exception:
        result['family'] = 'unknown'

    # `meta` device tracing avoids ever materializing real weights (safe for multi-billion
    # param models), but a minority of architectures call real-data ops (torch.unique,
    # Tensor.item()) from *config/init* logic rather than forward, which meta tensors can't
    # satisfy. Those are false negatives, not genuine export incompatibilities, so on any
    # meta-device failure we retry once for real on CPU before giving up.
    model = None
    try:
        with torch.device('meta'):
            model = timm.create_model(model_name, pretrained=False)
    except Exception:
        pass

    used_device = 'meta'
    if model is None:
        used_device = 'cpu'
        try:
            model = timm.create_model(model_name, pretrained=False)
        except Exception as e:
            result['status'] = 'create_failed'
            result['error'] = str(e)[:300]
            print(json.dumps(result))
            return

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    result['num_params'] = num_params
    result['weight_bytes'] = weight_bytes

    # Registry lookup, not a download: whether timm has anywhere to fetch real weights from
    # for this architecture, independent of whether this run traced it with real or random
    # ones. `select_models.py` needs this same fact to decide the release tier, so it lives
    # in exportlib rather than being derived twice.
    result['pretrained_tag'], _ = pretrained_info(model_name)

    # timm's own declared intent: fixed_input_size=False means the architecture (typically
    # global-pooled CNNs) is *designed* to accept other resolutions. This is a static claim
    # from the model's config, not a verified fact -- some archs mis-declare it, which is
    # exactly what the dynamic_export check below verifies empirically.
    try:
        fixed_input_size = model.default_cfg.get('fixed_input_size', None)
        result['resizable_cfg'] = None if fixed_input_size is None else not fixed_input_size
    except Exception:
        result['resizable_cfg'] = None

    # Whatever docstring the entrypoint carries (often a paper reference, sometimes nothing) --
    # inconsistent across the ~1300 architectures, but still more useful than a blank cell.
    try:
        doc = model_entrypoint(model_name).__doc__
        paper_ids = model.default_cfg.get('paper_ids')
        result['description'] = fmt_description(doc, paper_ids)
    except Exception:
        result['description'] = ''

    # timm's canonical, model-specific preprocessing recipe (resize/crop/normalize) -- the
    # same config `timm.data.create_transform()` consumes to build the actual transform.
    try:
        from timm.data import resolve_data_config
        result['preprocessing'] = fmt_preprocessing(resolve_data_config(model=model))
    except Exception:
        result['preprocessing'] = ''

    def try_export(device):
        input_size = resolved_input_size(model.default_cfg, max_res)
        if device == 'meta':
            example = torch.empty(1, *input_size, device='meta')
        else:
            example = torch.randn(1, *input_size)
        with flop_counter.FlopCounterMode(display=False) as fc:
            ep = torch.export.export(model, (example,))
        return ep, list(input_size), fc.get_total_flops()

    def try_dynamic_export(device, input_size):
        # Ground truth for "does this architecture actually tolerate other resolutions":
        # attempt export again with H/W marked dynamic. Catches cases where the cfg claims
        # resizability but some op (patch-embed divisibility, a fixed positional-embedding
        # table, a baked-in reshape) specializes to the concrete size anyway.
        if device == 'meta':
            example = torch.empty(1, *input_size, device='meta')
        else:
            example = torch.randn(1, *input_size)
        dynamic_shapes = ({2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC},)
        ep = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)
        return describe_dynamic_shapes(ep)

    export_device = None
    exported = None
    try:
        exported, input_size, flops = try_export(used_device)
        result['input_size'] = input_size
        result['resolution'] = f'{input_size[1]}x{input_size[2]}'
        result['status'] = 'ok'
        result['flops'] = flops
        export_device = used_device
    except Exception as e:
        export_error = e
        if used_device == 'meta':
            # Retry the export (rebuilding for real, since the current model's weights are
            # meta tensors) for real on CPU before giving up.
            try:
                model = timm.create_model(model_name, pretrained=False)
                model.eval()
                exported, input_size, flops = try_export('cpu')
                result['input_size'] = input_size
                result['resolution'] = f'{input_size[1]}x{input_size[2]}'
                result['status'] = 'ok'
                result['flops'] = flops
                export_device = 'cpu'
            except Exception as e2:
                result['status'] = 'export_failed'
                result['error'] = str(e2)[:300]
        else:
            result['status'] = 'export_failed'
            result['error'] = str(export_error)[:300]

    if collect and exported is not None:
        # The op cross-reference is harvested from the export we already paid for, one
        # decomposition later: torch.export hands back pre-dispatch ATen (conv2d,
        # batch_norm, linear), while what a PT2 backend actually lowers is core ATen
        # (convolution, _native_batch_norm_legit_no_training, addmm).
        try:
            with time_budget(ops_timeout):
                result['ops'], result['op_schemas'] = collect_ops(exported.run_decompositions())
        except TimeoutError:
            result['ops'] = None
            result['ops_error'] = f'decomposition exceeded {ops_timeout}s'
        except Exception as e:
            result['ops'] = None
            result['ops_error'] = str(e)[:200]

    if export_device is not None:
        # Guard-solving for dynamic H/W is cheap (~seconds) for most architectures but, for a
        # minority (windowed/halo attention with shape-dependent padding math), effectively
        # never converges -- observed running 400s+ without finishing. Bound it on its own
        # short budget so a slow/hanging dynamic check can never cost us the already-computed
        # static results (params, FLOPs, preprocessing, description).
        #
        # Models on the exclusion list are ones whose answer is known to depend on how fast
        # the machine is rather than on the architecture; they are reported as `excluded`
        # instead of being measured, which is both stable and honest about not knowing.
        if excluded:
            result['resizable_export'] = 'excluded'
        else:
            try:
                with time_budget(dynamic_timeout):
                    result['resizable_export'] = try_dynamic_export(export_device, input_size)
            except TimeoutError:
                result['resizable_export'] = 'timeout'
            except Exception:
                result['resizable_export'] = False

    print(json.dumps(result))


def fmt_params(n):
    if n is None:
        return ''
    if n >= 1e9:
        return f'{n / 1e9:.2f}B'
    return f'{n / 1e6:.1f}M'


def fmt_mb(n):
    if n is None:
        return ''
    return f'{n / (1024 * 1024):.1f}'


def fmt_gflops(flops):
    if flops is None:
        return None
    return flops / 1e9


def fmt_status(status):
    return '✅' if status == 'ok' else f'❌ {status}'


def parse_status(display):
    display = display.strip()
    if display == '✅':
        return 'ok'
    return display[1:].strip() if display.startswith('❌') else display


def fmt_bool(b):
    if b is None:
        return ''
    return '✅' if b else '❌'


def fmt_dynamic_export(v):
    """v is False/None/'timeout', or -- on success -- describe_dynamic_shapes()'s rendering
    of the exported graph's actual retained symbols and their real bounds, e.g.
    'H∈[2,∞] W∈[33,∞]', or 'H=W∈[64,∞]' if H and W collapsed onto
    one shared symbol (square-only resizing)."""
    if v is None:
        return ''
    if v is False:
        return '❌'
    return str(v)


def fmt_pretrained(tag):
    return tag if tag else '❌'


def _sanitize(s):
    return s.replace('|', '/').replace('\n', ' ').strip()


def fmt_description(doc, paper_ids):
    text = ''
    if doc:
        text = _sanitize(doc.strip().splitlines()[0])[:200]
    if paper_ids:
        suffix = f'({paper_ids})'
        text = f'{text} {suffix}' if text else suffix
    return text


def fmt_preprocessing(data_config):
    """Summarize timm's resolved per-model preprocessing recipe (the same config
    timm.data.create_transform() consumes): crop, interpolation, normalization. Deliberately
    excludes resolution -- that's reported in its own column, tied to the GFLOPs it produced,
    since for dynamic-shape-capable architectures it's a reference default, not a requirement."""
    if not data_config:
        return ''
    parts = []
    if data_config.get('crop_pct') is not None:
        parts.append(f"crop_pct={data_config['crop_pct']:g}")
    if data_config.get('crop_mode'):
        parts.append(data_config['crop_mode'])
    if data_config.get('interpolation'):
        parts.append(data_config['interpolation'])
    mean = data_config.get('mean')
    if mean:
        parts.append('mean=' + ','.join(f'{x:g}' for x in mean))
    std = data_config.get('std')
    if std:
        parts.append('std=' + ','.join(f'{x:g}' for x in std))
    return _sanitize(' '.join(parts))


def parse_bool(s):
    if s == '✅':
        return True
    if s == '❌':
        return False
    if s == 'timeout':
        return 'timeout'
    return None


def parse_dynamic_export(s):
    if not s:
        return None
    if s == '❌':
        return False
    return s  # 'timeout' or the verbatim descriptive success string


def parse_pretrained(s):
    return None if not s or s == '❌' else s


TABLE_ROW_RE = re.compile(
    r'^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<status>[^|]+?)\s*\|\s*(?P<params>[^|]*?)\s*\|'
    r'\s*(?P<weight>[^|]*?)\s*\|\s*(?P<pretrained>[^|]*?)\s*\|\s*(?P<resolution>[^|]*?)\s*\|'
    r'\s*(?P<gflops>[^|]*?)\s*\|'
    r'\s*(?P<resizable_cfg>[^|]*?)\s*\|\s*(?P<resizable_export>[^|]*?)\s*\|'
    r'\s*(?P<preprocessing>[^|]*?)\s*\|\s*(?P<description>[^|]*?)\s*\|\s*(?P<error>[^|]*?)\s*\|$'
)


def row_for_display(result):
    """Normalize a fresh worker result (raw num_params/weight_bytes/flops) into the
    pre-formatted display fields used both for rendering and for --resume round-tripping."""
    gflops = fmt_gflops(result.get('flops'))
    return {
        'name': result['name'],
        'family': result.get('family', 'unknown'),
        'status': result.get('status', 'unknown'),
        'params_str': fmt_params(result.get('num_params')),
        'weight_str': fmt_mb(result.get('weight_bytes')),
        'pretrained': result.get('pretrained_tag'),
        'resolution': result.get('resolution') or '',
        'gflops': gflops,
        'resizable_cfg': result.get('resizable_cfg'),
        'resizable_export': result.get('resizable_export'),
        'preprocessing': result.get('preprocessing') or '',
        'description': result.get('description') or '',
        'error': (result.get('error') or '').replace('|', '/').replace('\n', ' '),
    }


def parse_existing(path):
    """Parse a previously generated models.md back into {name: row} for --resume."""
    rows = {}
    if not os.path.exists(path):
        return rows
    family = 'unknown'
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('## '):
                family = line[3:].strip()
                continue
            m = TABLE_ROW_RE.match(line.strip())
            if not m or m.group('name') in ('variant',) or set(m.group('name')) <= {'-'}:
                continue
            d = m.groupdict()
            gflops = float(d['gflops']) if d['gflops'] else None
            rows[d['name']] = {
                'name': d['name'],
                'family': family,
                'status': parse_status(d['status']),
                'params_str': d['params'],
                'weight_str': d['weight'],
                'pretrained': parse_pretrained(d['pretrained']),
                'resolution': d['resolution'] or '',
                'gflops': gflops,
                'resizable_cfg': parse_bool(d['resizable_cfg']),
                'resizable_export': parse_dynamic_export(d['resizable_export']),
                'preprocessing': d['preprocessing'] or '',
                'description': d['description'] or '',
                'error': d['error'] or '',
            }
    return rows


def render_markdown(rows, timm_version, family_docs=None):
    family_docs = family_docs or {}
    by_family = {}
    for row in rows.values():
        by_family.setdefault(row.get('family', 'unknown'), []).append(row)

    ok = sum(1 for r in rows.values() if r.get('status') == 'ok')
    total = len(rows)

    lines = ['# timm model export report', '']
    lines.append(f'Generated by `scripts/export_report.py` against timm {timm_version}. '
                  f'{ok}/{total} variants exported successfully.')
    lines.append('')
    lines.append('✅/❌ mark `status` (export succeeded or not, failure kind alongside), `dynamic (cfg)`, '
                  '`dynamic (export)` (false only -- see below for its other values), and `pretrained` '
                  '(no fetchable weights). ')
    lines.append('`resolution` is the size the model was actually traced at to produce the reported GFLOPs -- '
                  'for architectures marked dynamic this is only a reference default, not a requirement. '
                  '`dynamic (cfg)` is timm\'s own declared intent (`fixed_input_size` in the model config); '
                  '`dynamic (export)` is verified by re-exporting with height/width marked dynamic via '
                  '`torch.export.Dim`, then reading the real free symbols and bounds the exported graph '
                  'actually retained (`H∈[2,∞] W∈[33,∞]`; `H=W∈[64,∞]` if H and W collapsed onto one '
                  'shared symbol, i.e. only square resizes work; `timeout` means guard-solving for dynamic '
                  'shapes did not converge within budget, mostly windowed/halo-attention architectures; '
                  '`excluded` means the check is skipped because its outcome was found to depend on how '
                  'fast the machine is rather than on the architecture -- see `export-exclusions.yaml`). '
                  '`preprocessing` is timm\'s resolved per-model recipe (the same config '
                  '`timm.data.create_transform()` consumes): crop, interpolation, normalization. '
                  '`pretrained` is the tag of the weights timm can actually fetch for this variant '
                  '(from its default `PretrainedCfg`, independent of whether this run traced it with '
                  'real or random weights), or ❌ if it has none -- the same fact that decides '
                  '`models-selected.yaml`\'s `release` flag. '
                  '`description` is the model entrypoint\'s docstring where available (inconsistent across '
                  'architectures, but better than nothing).')
    lines.append('')

    def sort_key(r):
        # Round to the precision actually rendered below: an unrounded key orders ties
        # that display identically by their invisible sub-hundredth difference, which
        # can flip between machines (e.g. SDPA backend dispatch nudges an attention
        # model's flop count) -- reordering rows in CI with nothing visible to justify
        # the diff. Rounding first makes the order depend only on what's on the page.
        gflops = r.get('gflops')
        return (gflops is None, round(gflops, 2) if gflops is not None else 0.0, r['name'])

    for family in sorted(by_family):
        lines.append(f'## {family}')
        lines.append('')
        doc = family_docs.get(family)
        if doc:
            lines.append(f'_{doc}_')
            lines.append('')
        lines.append('| variant | status | params | weight (MB) | pretrained | resolution | GFLOPs | '
                      'dynamic (cfg) | dynamic (export) | preprocessing | description | error |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')

        for row in sorted(by_family[family], key=sort_key):
            gflops = row.get('gflops')
            lines.append(
                f"| {row['name']} | {fmt_status(row['status'])} | {row['params_str']} | "
                f"{row['weight_str']} | {fmt_pretrained(row.get('pretrained'))} | {row['resolution']} | "
                f"{f'{gflops:.2f}' if gflops is not None else ''} | "
                f"{fmt_bool(row.get('resizable_cfg'))} | {fmt_dynamic_export(row.get('resizable_export'))} | "
                f"{row['preprocessing']} | {row['description']} | {row['error']} |"
            )
        lines.append('')

    return '\n'.join(lines)


def short_op(op):
    """`aten.convolution.default` -> `convolution.default`. The overload suffix stays:
    different overloads are different operators to whoever has to implement them."""
    return op[len('aten.'):] if op.startswith('aten.') else op


def full_op(op):
    """Inverse of short_op(). An operator name is `namespace.name.overload`, so a key with
    a namespace still on it (anything but aten's) keeps exactly two dots and is left alone."""
    return op if op.count('.') >= 2 else f'aten.{op}'


def build_op_index(ops_by_model):
    """({op: {canonical config: id}}, {op: [(id, config)]}) over every model in the run.

    Ids are per-op and positional: 1..N over that op's configurations in canonical sort
    order, which makes them deterministic for a given run but not stable across runs that
    discover a new configuration sorting ahead of an existing one.
    """
    by_op = {}
    for ops in ops_by_model.values():
        for op, config, _count in ops:
            by_op.setdefault(op, {}).setdefault(canonical_config(config), config)

    index, catalog = {}, {}
    for op, configs in by_op.items():
        ordered = sorted(configs)
        index[op] = {canonical: i for i, canonical in enumerate(ordered, 1)}
        catalog[op] = [(i, configs[canonical]) for i, canonical in enumerate(ordered, 1)]
    return index, catalog


def op_usage(ops_by_model, families):
    """{op: {config id: {'models': set, 'families': set, 'nodes': n, 'example': name}}}.

    Models and families are kept as sets rather than counters because ops.md reports both
    per configuration and rolled up per operator, and a variant using four configurations
    of `convolution` is still one variant at the operator level.
    """
    index, _ = build_op_index(ops_by_model)
    usage = {}
    for name in sorted(ops_by_model):
        family = families.get(name, 'unknown')
        for op, config, count in ops_by_model[name]:
            cell = usage.setdefault(op, {}).setdefault(
                index[op][canonical_config(config)],
                {'models': set(), 'families': set(), 'nodes': 0, 'example': name},
            )
            cell['models'].add(name)
            cell['families'].add(family)
            cell['nodes'] += count
    return usage


class _FlowMap(dict):
    """A mapping rendered inline (`{stride: [1, 1], groups: 1}`) rather than as a block."""


class _OpsDumper(yaml.SafeDumper):
    pass


_OpsDumper.add_representer(
    _FlowMap,
    lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True),
)


def render_ops_yaml(ops_by_model, op_schemas, skipped, timm_version, torch_version):
    """The cross-reference itself: an operator catalog plus a sparse models × operations
    matrix whose cells are {configuration id: node count}.

    Flow-style mappings and an effectively unlimited line width keep one model-operation
    pair (and one configuration) to a line, which is what makes ~1300 models greppable and
    diffable; PyYAML's default block style and 80-column wrapping do neither.
    """
    index, catalog = build_op_index(ops_by_model)

    ops = {}
    for op in sorted(catalog):
        ops[short_op(op)] = {
            'schema': op_schemas.get(op, ''),
            'configs': {i: _FlowMap(config) for i, config in catalog[op]},
        }

    models = {}
    for name in sorted(ops_by_model):
        cells = {}
        for op, config, count in ops_by_model[name]:
            cells.setdefault(short_op(op), {})[index[op][canonical_config(config)]] = count
        models[name] = {op: _FlowMap(sorted(cell.items())) for op, cell in sorted(cells.items())}

    document = {'ops': ops, 'models': models}
    if skipped:
        document['skipped'] = {name: skipped[name] for name in sorted(skipped)}

    header = [
        '# timm x aten cross-reference -- generated by scripts/export_report.py, do not edit.',
        '#',
        f'# ir: core ATen (torch.export + run_decompositions)   timm: {timm_version}   torch: {torch_version}',
        '#',
        '# ops.<op>.configs[<id>] -- one distinct configuration of that operator seen across the',
        '#   zoo: its non-Tensor schema arguments, plus out_dtype/out_rank read from the graph\'s',
        '#   own shape metadata (and kernel/depthwise for convolution, which the argument list',
        '#   does not carry but a backend very much cares about).',
        '# models.<variant>.<op> -- {configuration id: number of nodes using it}.',
        '# skipped -- models that exported but whose graph could not be decomposed.',
        '#',
        '# SymInt arguments are tensor extents that scale with the input resolution, so they are',
        '# recorded as arity ("[*4]", or "*" for a scalar) rather than verbatim, except on',
        '# ' + ', '.join(sorted(short_op(op) for op in SYMINT_LITERAL_OPS)) + ',',
        '# where they are architectural knobs. Configuration ids are positional within an op, so',
        '# they can shift between runs when a newly seen configuration sorts ahead of an old one.',
        '',
    ]
    body = yaml.dump(document, Dumper=_OpsDumper, sort_keys=False, width=10 ** 6, default_flow_style=False)
    return '\n'.join(header) + body


def fmt_config(config):
    """A configuration as compact one-line text for the markdown digest."""
    parts = []
    for key, value in config.items():
        if isinstance(value, list):
            rendered = '[' + ','.join(str(v) for v in value) + ']'
        elif isinstance(value, bool):
            rendered = 'true' if value else 'false'
        elif value is None:
            rendered = 'none'
        else:
            rendered = str(value)
        parts.append(f'{key}={rendered}')
    return _sanitize(' '.join(parts)) or '(no arguments)'


def render_ops_md(ops_by_model, op_schemas, families, skipped, timm_version, torch_version):
    """Op-major digest of the cross-reference: for each operator, which configurations the
    zoo needs and how many variants need each. The model axis is collapsed to counts --
    the per-variant detail lives in ops.yaml, which is the machine-readable artifact."""
    _, catalog = build_op_index(ops_by_model)
    usage = op_usage(ops_by_model, families)

    total_configs = sum(len(configs) for configs in catalog.values())

    def op_models(op):
        return set().union(*(c['models'] for c in usage[op].values()))

    ordered_ops = sorted(catalog, key=lambda op: (-len(op_models(op)), op))

    lines = ['# timm aten operator cross-reference', '']
    lines.append(f'Generated by `scripts/export_report.py` against timm {timm_version} and torch '
                 f'{torch_version}. {len(ops_by_model)} exported variants use {len(catalog)} distinct '
                 f'core ATen operators in {total_configs} distinct configurations.')
    lines.append('')
    lines.append('Operators are read from the graph `torch.export` produces after '
                 '`run_decompositions()` -- core ATen, what a PT2 backend actually lowers, rather than '
                 'the pre-dispatch `conv2d`/`batch_norm`/`linear` the exporter hands back first. '
                 'A *configuration* is an operator\'s non-Tensor schema arguments plus `out_dtype` '
                 'and `out_rank` taken from the graph\'s shape metadata, and for convolutions the '
                 '`kernel` size and whether it is `depthwise`. SymInt arguments are tensor extents that scale '
                 'with input resolution, so they are recorded as arity (`size=[*4]`) rather than '
                 'verbatim, except on ' +
                 ', '.join(f'`{short_op(op)}`' for op in sorted(SYMINT_LITERAL_OPS)) +
                 ', where they are architectural knobs. `models` counts variants using that '
                 'configuration, `nodes` the total call sites across them. The full per-variant '
                 'matrix is in [`ops.yaml`](ops.yaml); ids here are that file\'s ids.')
    lines.append('')

    lines.append('## summary')
    lines.append('')
    lines.append('| op | configs | models | families | nodes |')
    lines.append('|---|---|---|---|---|')
    for op in ordered_ops:
        cells = usage[op].values()
        lines.append(f'| {short_op(op)} | {len(catalog[op])} | {len(op_models(op))} | '
                     f'{len(set().union(*(c["families"] for c in cells)))} | '
                     f'{sum(c["nodes"] for c in cells)} |')
    lines.append('')

    lines.append('## configurations')
    lines.append('')
    for op in ordered_ops:
        lines.append(f'### {short_op(op)}')
        lines.append('')
        schema = op_schemas.get(op)
        if schema:
            lines.append(f'`{schema}`')
            lines.append('')
        lines.append('| id | configuration | models | families | nodes | e.g. |')
        lines.append('|---|---|---|---|---|---|')
        for i, config in sorted(catalog[op], key=lambda c: (-len(usage[op][c[0]]['models']), c[0])):
            cell = usage[op][i]
            lines.append(f'| {i} | {fmt_config(config)} | {len(cell["models"])} | {len(cell["families"])} | '
                         f'{cell["nodes"]} | {cell["example"]} |')
        lines.append('')

    if skipped:
        lines.append('## skipped')
        lines.append('')
        lines.append('Variants that exported but whose graph could not be decomposed, so they '
                     'contribute to neither this file nor `ops.yaml`.')
        lines.append('')
        lines.append('| variant | reason |')
        lines.append('|---|---|')
        for name in sorted(skipped):
            lines.append(f'| {name} | {_sanitize(skipped[name])} |')
        lines.append('')

    return '\n'.join(lines)


def parse_exclusions(path):
    """{model: reason} from the exclusion file, or {} if there is none."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        document = yaml.load(f, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader)) or {}
    return {name: str(entry) for name, entry in (document.get('models') or {}).items()}


def render_exclusions(rows, dynamic_timeout):
    """The models whose dynamic-shape check ran out of budget, as a file to commit.

    Whether one of these converges is a fact about how fast the machine is, not about the
    architecture: on this box `tf_efficientnet_b0` settles in 17s, on a CI runner the same
    check blows a 60s budget. Recording that verdict in models.md makes the report
    non-reproducible across machines, so instead the slow ones are listed here once,
    deliberately, and reported as `excluded` from then on.

    Generated at the default budget and consumed by runs at any budget: a slower machine
    raises its own timeouts (see the Makefile's CI target) rather than re-deriving the
    list, so both sides exclude exactly the same models.
    """
    excluded = sorted(name for name, row in rows.items()
                      if row.get('resizable_export') == 'timeout')
    lines = [
        '# Models excluded from the dynamic-shape (`dynamic (export)`) check.',
        '#',
        '# Generated by `make report.exclusions`, do not edit by hand -- and regenerate it',
        f'# rather than adding entries, so every listed model was measured the same way.',
        f'# These exceeded the {dynamic_timeout}s guard-solving budget, which makes their',
        '# result depend on machine speed; `make report` reports them as `excluded`.',
        '',
        'models:',
    ]
    for name in excluded:
        lines.append(f'  {name}: guard solving exceeded the {dynamic_timeout}s budget')
    lines.append('')
    return '\n'.join(lines)


def parse_existing_ops(path):
    """Parse a previously generated ops.yaml back into
    ({name: [[op, config, count], ...]}, {op: schema}, {name: reason}) for --resume."""
    if not os.path.exists(path):
        return {}, {}, {}
    with open(path) as f:
        document = yaml.load(f, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader)) or {}

    catalog = document.get('ops') or {}
    schemas = {}
    configs = {}  # short op -> {id: config}
    for op, entry in catalog.items():
        schemas[full_op(op)] = entry.get('schema', '')
        configs[op] = entry.get('configs') or {}

    ops_by_model = {}
    for name, cells in (document.get('models') or {}).items():
        ops = []
        for op, counts in cells.items():
            for config_id, count in counts.items():
                config = configs.get(op, {}).get(config_id)
                if config is None:
                    continue  # catalog/matrix disagree -- treat the model as incomplete
                ops.append([full_op(op), dict(config), count])
        ops_by_model[name] = ops

    return ops_by_model, schemas, dict(document.get('skipped') or {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter', default='', help='fnmatch glob(s), comma-separated, to select model names')
    parser.add_argument('--exclude', default='', help='fnmatch glob(s), comma-separated, to exclude model names')
    parser.add_argument('--limit', type=int, default=None, help='only process the first N models')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                         help='parallel model subprocesses (default: one per available core). Each holds '
                              'a whole model, and the CPU-fallback path materializes real weights, so on '
                              'a machine with many cores but little RAM this is the knob to turn down -- '
                              'an OOM-killed worker is recorded as a crashed model, not a hard failure')
    parser.add_argument('--timeout', type=float, default=120.0, help='per-model subprocess timeout, seconds')
    parser.add_argument('--dynamic-timeout', type=int, default=60,
                         help='budget (seconds) for the dynamic-shapes export sub-check; some architectures '
                              '(windowed/halo attention) never converge on it, so it is bounded independently '
                              'of --timeout to avoid losing already-computed static results')
    parser.add_argument('--ops-timeout', type=int, default=120,
                         help='budget (seconds) for decomposing the exported graph into core ATen for the '
                              'op cross-reference; bounded independently of --timeout for the same reason '
                              'as --dynamic-timeout')
    parser.add_argument('--max-res', type=int, default=MAX_RES, help='cap input resolution used for export')
    parser.add_argument('--output', default=None, help='output models.md path (default: repo-root models.md)')
    parser.add_argument('--ops-output', default=None,
                         help='output ops.yaml path, the models x operations cross-reference '
                              '(default: repo-root ops.yaml)')
    parser.add_argument('--ops-md', default=None,
                         help='output ops.md path, the op-major digest of the cross-reference '
                              '(default: repo-root ops.md)')
    parser.add_argument('--no-ops', action='store_true',
                         help='skip op collection entirely and write only models.md')
    parser.add_argument('--exclusions', default=None,
                         help='YAML list of models to report as `excluded` instead of running the '
                              'dynamic-shapes check on them (default: repo-root export-exclusions.yaml '
                              'if it exists)')
    parser.add_argument('--write-exclusions', action='store_true',
                         help='regenerate the exclusion file from this run: every model whose dynamic '
                              'check hit --dynamic-timeout is listed. Run locally, without --exclusions, '
                              'so that every model is actually measured')
    parser.add_argument('--resume', action='store_true', help='skip models already recorded in --output')
    parser.add_argument('--checkpoint-every', type=int, default=25, help='rewrite output every N completions')
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--excluded', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker, args.max_res, args.dynamic_timeout, args.ops_timeout,
                    not args.no_ops, args.excluded)
        return

    import importlib
    import timm
    import torch
    from timm.models import model_entrypoint

    torch_version = torch.__version__

    def family_of(name):
        try:
            return model_entrypoint(name).__module__.rsplit('.', 1)[-1]
        except Exception:
            return 'unknown'

    # Every family module (timm/models/<family>.py) carries a module-level docstring --
    # usually a title plus paper references, e.g. "The EfficientNet Family in PyTorch" --
    # a much richer description than any single variant's own (often blank) docstring.
    family_docs = {}
    for module_name in timm.list_modules():
        try:
            doc = importlib.import_module(f'timm.models.{module_name}').__doc__
        except Exception:
            doc = None
        if doc and doc.strip():
            family_docs[module_name] = _sanitize(doc.strip().splitlines()[0])[:200]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = args.output or os.path.join(repo_root, 'models.md')
    ops_path = args.ops_output or os.path.join(repo_root, 'ops.yaml')
    ops_md_path = args.ops_md or os.path.join(repo_root, 'ops.md')
    exclusions_path = args.exclusions or os.path.join(repo_root, 'export-exclusions.yaml')
    collect = not args.no_ops

    # Regenerating the list means measuring every model, so the two are mutually exclusive.
    exclusions = {} if args.write_exclusions else parse_exclusions(exclusions_path)
    if exclusions:
        print(f'Excluding the dynamic-shapes check for {len(exclusions)} models '
              f'({os.path.basename(exclusions_path)})', file=sys.stderr)

    names = timm.list_models(filter=globs(args.filter), exclude_filters=globs(args.exclude), pretrained=False)
    if args.limit:
        names = names[:args.limit]

    rows = {}
    ops_by_model, op_schemas, ops_skipped = {}, {}, {}
    if args.resume:
        rows = parse_existing(output_path)
        if collect:
            ops_by_model, op_schemas, ops_skipped = parse_existing_ops(ops_path)

        def done(name):
            if name not in rows or rows[name].get('status') not in ('ok', 'export_failed', 'create_failed'):
                return False
            # A model that exported but is missing from the cross-reference has ops still to
            # collect, so it is not done -- which is also what makes the first run after
            # adding ops.yaml re-export everything, rather than emitting an empty matrix.
            if collect and rows[name].get('status') == 'ok':
                return name in ops_by_model or name in ops_skipped
            return True

        names = [n for n in names if not done(n)]

    total = len(names)
    print(f'Running {total} models with {args.workers} workers (timeout={args.timeout}s each)...', file=sys.stderr)

    def write_outputs():
        with open(output_path, 'w') as f:
            f.write(render_markdown(rows, timm.__version__, family_docs))
        if not collect:
            return
        families = {name: rows[name].get('family', 'unknown') for name in rows}
        with open(ops_path, 'w') as f:
            f.write(render_ops_yaml(ops_by_model, op_schemas, ops_skipped, timm.__version__, torch_version))
        with open(ops_md_path, 'w') as f:
            f.write(render_ops_md(ops_by_model, op_schemas, families, ops_skipped,
                                  timm.__version__, torch_version))

    completed = 0
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_worker, name, args.max_res, args.timeout, args.dynamic_timeout,
                               args.ops_timeout, collect, name in exclusions): name
                   for name in names}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {'name': name, 'status': 'crashed', 'error': str(e)[:300]}
            if not result.get('family'):
                result['family'] = family_of(name)
            rows[result['name']] = row_for_display(result)
            if collect:
                ops_by_model.pop(name, None)
                ops_skipped.pop(name, None)
                if result.get('ops'):
                    ops_by_model[name] = result['ops']
                    op_schemas.update(result.get('op_schemas') or {})
                elif result.get('status') == 'ok':
                    ops_skipped[name] = result.get('ops_error') or 'no ops collected'
            completed += 1
            elapsed = time.monotonic() - start
            print(f'[{completed}/{total}] {name}: {result["status"]} ({elapsed:.0f}s elapsed)', file=sys.stderr)

            if completed % args.checkpoint_every == 0:
                write_outputs()

    if args.write_exclusions:
        with open(exclusions_path, 'w') as f:
            f.write(render_exclusions(rows, args.dynamic_timeout))
        # The models that just timed out are exactly the ones every later run will skip, so
        # mark them as such here too: one generating run leaves a tree consistent with what
        # `make report` would produce next, instead of needing a second full pass.
        newly_excluded = [name for name, row in rows.items() if row.get('resizable_export') == 'timeout']
        for name in newly_excluded:
            rows[name]['resizable_export'] = 'excluded'
        print(f'Wrote {exclusions_path} ({len(newly_excluded)} models)', file=sys.stderr)

    write_outputs()

    print(f'Wrote {output_path}' + ('' if not collect else f', {ops_path}, {ops_md_path}'), file=sys.stderr)


if __name__ == '__main__':
    main()
