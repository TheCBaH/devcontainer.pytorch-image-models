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
import json
import os
import re
import sys
import time

import yaml

from exportlib import MAX_RES, pretrained_info, resolved_input_size
from pt2_export_core.catalog import ATEN, CORE_BACKENDS, FUNC, core, parse_existing_ops
from pt2_export_core.catalog import render_ops_md as _render_ops_md
from pt2_export_core.catalog import render_ops_yaml as _render_ops_yaml
from pt2_export_core.exclusions import parse_exclusions, render_exclusions as _render_exclusions
from pt2_export_core.harness import cpu_count, globs, run_worker as _run_worker
from pt2_export_core.markdown import heading_anchor
from pt2_export_core.opgraph import collect_ops, describe_dynamic_shapes as _describe_dynamic_shapes, time_budget


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
    return _describe_dynamic_shapes(ep, axis_names=_AXIS_NAMES)


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
    result['pretrained_tag'], _, _ = pretrained_info(model_name)

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

    # Which device the graph was traced on, and so which core ATen cross-reference the model
    # belongs in: decomposition runs after dispatch, so a model that fell back to CPU was
    # lowered by different kernels than one traced on meta.
    result['device'] = export_device

    if collect and exported is not None:
        # All three dialects out of the one export already paid for, in order of how much they
        # rewrite. ATen first and from the program itself -- it is what torch.export hands back,
        # and what this repo publishes. The other two are the same AOTDispatcher retrace with
        # different decomposition tables: an empty one leaves only the functionalization the
        # retrace does unconditionally, the default one lowers all the way to core ATen.
        #
        # Collected independently so they fail independently: the retraces are the expensive
        # half, and a model that runs out of budget there still has an ATen graph worth keeping.
        result['ops'], result['op_schemas'], result['ops_error'] = {}, {}, {}
        for key, graph in (('aten', lambda: exported),
                           ('func', lambda: exported.run_decompositions(decomp_table={})),
                           ('core', lambda: exported.run_decompositions())):
            try:
                with time_budget(ops_timeout):
                    result['ops'][key], result['op_schemas'][key] = collect_ops(graph())
            except TimeoutError:
                result['ops_error'][key] = f'{key} op collection exceeded {ops_timeout}s'
            except Exception as e:
                result['ops_error'][key] = str(e)[:200]

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
    r'\s*(?P<gflops>[^|]*?)\s*\|\s*(?P<device>[^|]*?)\s*\|'
    r'\s*(?P<aten_nodes>[^|]*?)\s*\|\s*(?P<aten_ops>[^|]*?)\s*\|'
    r'\s*(?P<func_nodes>[^|]*?)\s*\|\s*(?P<func_ops>[^|]*?)\s*\|'
    r'\s*(?P<core_nodes>[^|]*?)\s*\|\s*(?P<core_ops>[^|]*?)\s*\|'
    r'\s*(?P<resizable_cfg>[^|]*?)\s*\|\s*(?P<resizable_export>[^|]*?)\s*\|'
    r'\s*(?P<preprocessing>[^|]*?)\s*\|\s*(?P<description>[^|]*?)\s*\|\s*(?P<error>[^|]*?)\s*\|$'
)


def fmt_count(n):
    return '' if n is None else str(n)


def parse_count(s):
    return int(s) if s else None


def dialect_counts(result, key):
    """(node count, distinct operator count) for one dialect of a worker result.

    Distinct *operators*, not configurations: configuration detail is what the cross-reference
    files exist for.
    """
    cells = (result.get('ops') or {}).get(key)
    if not cells:
        return None, None
    return sum(count for _, _, count in cells), len({op for op, _, _ in cells})


def row_for_display(result):
    """Normalize a fresh worker result (raw num_params/weight_bytes/flops) into the
    pre-formatted display fields used both for rendering and for --resume round-tripping."""
    gflops = fmt_gflops(result.get('flops'))
    aten_nodes, aten_ops = dialect_counts(result, 'aten')
    func_nodes, func_ops = dialect_counts(result, 'func')
    core_nodes, core_ops = dialect_counts(result, 'core')
    return {
        'name': result['name'],
        'family': result.get('family', 'unknown'),
        'status': result.get('status', 'unknown'),
        'params_str': fmt_params(result.get('num_params')),
        'weight_str': fmt_mb(result.get('weight_bytes')),
        'pretrained': result.get('pretrained_tag'),
        'resolution': result.get('resolution') or '',
        'gflops': gflops,
        'device': result.get('device') or '',
        'aten_nodes': aten_nodes,
        'aten_ops': aten_ops,
        'func_nodes': func_nodes,
        'func_ops': func_ops,
        'core_nodes': core_nodes,
        'core_ops': core_ops,
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
                'device': d['device'] or '',
                'aten_nodes': parse_count(d['aten_nodes']),
                'aten_ops': parse_count(d['aten_ops']),
                'func_nodes': parse_count(d['func_nodes']),
                'func_ops': parse_count(d['func_ops']),
                'core_nodes': parse_count(d['core_nodes']),
                'core_ops': parse_count(d['core_ops']),
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
    lines.append('`aten nodes`/`aten ops` count the graph `torch.export` hands back -- the ATen dialect, '
                  'with `conv2d`/`linear`/`layer_norm`/`scaled_dot_product_attention` whole -- '
                  '`func nodes`/`func ops` the same graph functionalized but not decomposed '
                  '(`run_decompositions(decomp_table={})`), and `core nodes`/`core ops` the same graph '
                  'decomposed all the way (`run_decompositions()`). `ops` is distinct operators, `nodes` '
                  'total call sites. The three are a progression in how much has been rewritten, not in '
                  'size: functionalizing costs a handful of nodes and usually no new operators at all, '
                  'while decomposition trades a few composite operators for many primitive ones '
                  '(`vit_tiny_patch16_224`: 227 nodes over 15 operators, 239 over 15, then 696 over 21). '
                  'No two of the operator sets contain each other. The per-variant matrices are '
                  '`ops-aten.yaml`, `ops-func.yaml` and `ops-core-<device>.yaml`. `device` is where the '
                  'model was traced -- `meta` normally, `cpu` for architectures meta cannot build -- and '
                  'so which core cross-reference it is in, that decomposition being backend-specific; '
                  'the other two dialects are written whole, being fixed before dispatch. ')
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

    # Anchors are minted in the order the headings are emitted, since that is what decides
    # GitHub's numeric suffix on any repeated slug.
    anchors = {}
    index_anchor = heading_anchor('families', anchors)
    family_anchor = {family: heading_anchor(family, anchors) for family in sorted(by_family)}

    def gflops_range(family_rows):
        """Span of the traced GFLOPs across a family -- the one number that separates a family's
        variants from each other, and so what a reader scanning the index is choosing between."""
        values = sorted(r['gflops'] for r in family_rows if r.get('gflops') is not None)
        if not values:
            return ''
        low, high = f'{values[0]:.2f}', f'{values[-1]:.2f}'
        return low if low == high else f'{low}-{high}'

    lines.append('## families')
    lines.append('')
    lines.append(f'{len(by_family)} architecture families, one table each below; the name links '
                  'to it. `variants` is how many this report covers, `exported` how many of those '
                  '`torch.export` accepted, `pretrained` how many have weights timm can fetch, '
                  'and `GFLOPs` the range over the exported ones at their traced resolution.')
    lines.append('')
    lines.append('| family | variants | exported | pretrained | GFLOPs |')
    lines.append('|---|---|---|---|---|')
    for family in sorted(by_family):
        family_rows = by_family[family]
        lines.append(
            f'| [{family}](#{family_anchor[family]}) | {len(family_rows)} | '
            f"{sum(1 for r in family_rows if r.get('status') == 'ok')} | "
            f"{sum(1 for r in family_rows if r.get('pretrained'))} | "
            f'{gflops_range(family_rows)} |'
        )
    lines.append('')

    for family in sorted(by_family):
        lines.append(f'## {family}')
        lines.append('')
        lines.append(f'[↑ families](#{index_anchor})')
        lines.append('')
        doc = family_docs.get(family)
        if doc:
            lines.append(f'_{doc}_')
            lines.append('')
        lines.append('| variant | status | params | weight (MB) | pretrained | resolution | GFLOPs | '
                      'device | aten nodes | aten ops | func nodes | func ops | core nodes | core ops | '
                      'dynamic (cfg) | dynamic (export) | preprocessing | description | error |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')

        for row in sorted(by_family[family], key=sort_key):
            gflops = row.get('gflops')
            lines.append(
                f"| {row['name']} | {fmt_status(row['status'])} | {row['params_str']} | "
                f"{row['weight_str']} | {fmt_pretrained(row.get('pretrained'))} | {row['resolution']} | "
                f"{f'{gflops:.2f}' if gflops is not None else ''} | {row.get('device') or ''} | "
                f"{fmt_count(row.get('aten_nodes'))} | {fmt_count(row.get('aten_ops'))} | "
                f"{fmt_count(row.get('func_nodes'))} | {fmt_count(row.get('func_ops'))} | "
                f"{fmt_count(row.get('core_nodes'))} | {fmt_count(row.get('core_ops'))} | "
                f"{fmt_bool(row.get('resizable_cfg'))} | {fmt_dynamic_export(row.get('resizable_export'))} | "
                f"{row['preprocessing']} | {row['description']} | {row['error']} |"
            )
        lines.append('')

    return '\n'.join(lines)


def render_ops_yaml(ops_by_model, op_schemas, skipped, dialect, timm_version, torch_version):
    return _render_ops_yaml(ops_by_model, op_schemas, skipped, dialect, 'timm', timm_version, torch_version)


def render_ops_md(ops_by_model, op_schemas, families, skipped, dialect, timm_version, torch_version):
    return _render_ops_md(ops_by_model, op_schemas, families, skipped, dialect, 'timm',
                          timm_version, torch_version)


def render_exclusions(rows, dynamic_timeout):
    excluded = sorted(name for name, row in rows.items()
                      if row.get('resizable_export') == 'timeout')
    return _render_exclusions(excluded, dynamic_timeout)


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
                         help='budget (seconds) for collecting one dialect of the op cross-reference '
                              'from an exported graph -- the core ATen half pays for a decomposition, '
                              'which is the slow part; bounded independently of --timeout for the same '
                              'reason as --dynamic-timeout')
    parser.add_argument('--max-res', type=int, default=MAX_RES, help='cap input resolution used for export')
    parser.add_argument('--output', default=None, help='output models.md path (default: repo-root models.md)')
    parser.add_argument('--ops-aten-output', default=None,
                         help='output ops-aten.yaml path, the models x operations cross-reference of the '
                              'graph torch.export hands back (default: repo-root ops-aten.yaml)')
    parser.add_argument('--ops-aten-md', default=None,
                         help='output ops-aten.md path, the op-major digest of that cross-reference '
                              '(default: repo-root ops-aten.md)')
    parser.add_argument('--ops-func-output', default=None,
                         help='output ops-func.yaml path, the same cross-reference of the graph '
                              'run_decompositions(decomp_table={}) produces -- functionalized, but '
                              'otherwise undecomposed (default: repo-root ops-func.yaml)')
    parser.add_argument('--ops-func-md', default=None,
                         help='output ops-func.md path, the op-major digest of that cross-reference '
                              '(default: repo-root ops-func.md)')
    parser.add_argument('--ops-core-prefix', default=None,
                         help='path prefix for the core ATen cross-references; each backend gets its own '
                              '<prefix>-<backend>.yaml and .md, because that decomposition depends on the '
                              'device the model was traced on (default: repo-root ops-core)')
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
    aten_yaml_path = args.ops_aten_output or os.path.join(repo_root, f'{ATEN.stem}.yaml')
    aten_md_path = args.ops_aten_md or os.path.join(repo_root, f'{ATEN.stem}.md')
    func_yaml_path = args.ops_func_output or os.path.join(repo_root, f'{FUNC.stem}.yaml')
    func_md_path = args.ops_func_md or os.path.join(repo_root, f'{FUNC.stem}.md')
    core_prefix = args.ops_core_prefix or os.path.join(repo_root, 'ops-core')
    collect = not args.no_ops

    def core_paths(backend):
        return f'{core_prefix}-{backend}.yaml', f'{core_prefix}-{backend}.md'

    exclusions_path = args.exclusions or os.path.join(repo_root, 'export-exclusions.yaml')

    # Regenerating the list means measuring every model, so the two are mutually exclusive.
    exclusions = {} if args.write_exclusions else parse_exclusions(exclusions_path)
    if exclusions:
        print(f'Excluding the dynamic-shapes check for {len(exclusions)} models '
              f'({os.path.basename(exclusions_path)})', file=sys.stderr)

    names = timm.list_models(filter=globs(args.filter), exclude_filters=globs(args.exclude), pretrained=False)
    if args.limit:
        names = names[:args.limit]

    rows = {}
    # One matrix per dialect. The core one is kept whole here and partitioned by export device
    # only when it is written out: a model's backend is a property of the row, so there is
    # nothing to gain from carrying the split through the run itself.
    ops_by_model = {'aten': {}, 'func': {}, 'core': {}}
    op_schemas = {'aten': {}, 'func': {}, 'core': {}}
    ops_skipped = {'aten': {}, 'func': {}, 'core': {}}
    if args.resume:
        rows = parse_existing(output_path)
        if collect:
            ops_by_model['aten'], op_schemas['aten'], ops_skipped['aten'] = \
                parse_existing_ops(aten_yaml_path)
            ops_by_model['func'], op_schemas['func'], ops_skipped['func'] = \
                parse_existing_ops(func_yaml_path)
            for backend in CORE_BACKENDS:
                matrix, schemas, skipped = parse_existing_ops(core_paths(backend)[0])
                ops_by_model['core'].update(matrix)
                op_schemas['core'].update(schemas)
                ops_skipped['core'].update(skipped)

        def done(name):
            if name not in rows or rows[name].get('status') not in ('ok', 'export_failed', 'create_failed'):
                return False
            # A model that exported but is missing from either cross-reference has ops still to
            # collect, so it is not done -- which is also what makes the first run after a new
            # dialect is added re-export everything, rather than emitting an empty matrix.
            if collect and rows[name].get('status') == 'ok':
                return all(name in ops_by_model[key] or name in ops_skipped[key]
                           for key in ops_by_model)
            return True

        names = [n for n in names if not done(n)]

    total = len(names)
    print(f'Running {total} models with {args.workers} workers (timeout={args.timeout}s each)...', file=sys.stderr)

    def write_ops(dialect, yaml_path, md_path, matrix, schemas, skipped, families):
        with open(yaml_path, 'w') as f:
            f.write(render_ops_yaml(matrix, schemas, skipped, dialect, timm.__version__, torch_version))
        with open(md_path, 'w') as f:
            f.write(render_ops_md(matrix, schemas, families, skipped, dialect,
                                  timm.__version__, torch_version))

    def write_outputs():
        with open(output_path, 'w') as f:
            f.write(render_markdown(rows, timm.__version__, family_docs))
        if not collect:
            return
        families = {name: rows[name].get('family', 'unknown') for name in rows}
        write_ops(ATEN, aten_yaml_path, aten_md_path, ops_by_model['aten'], op_schemas['aten'],
                  ops_skipped['aten'], families)
        # Functionalization happens in the retrace, before dispatch, so this one is written whole
        # like the ATen cross-reference rather than split the way the core ones below are.
        write_ops(FUNC, func_yaml_path, func_md_path, ops_by_model['func'], op_schemas['func'],
                  ops_skipped['func'], families)

        # Split by the device each model was traced on, since that is what decided its
        # decomposition. CORE_BACKENDS is written even when empty: a missing file would leave a
        # reader guessing whether the sweep or the file was incomplete.
        seen = {row.get('device') for row in rows.values() if row.get('device')}
        for backend in list(CORE_BACKENDS) + sorted(seen - set(CORE_BACKENDS)):
            on_backend = {name for name, row in rows.items() if row.get('device') == backend}
            yaml_path, md_path = core_paths(backend)
            write_ops(core(backend), yaml_path, md_path,
                      {n: v for n, v in ops_by_model['core'].items() if n in on_backend},
                      op_schemas['core'],
                      {n: v for n, v in ops_skipped['core'].items() if n in on_backend},
                      families)

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
                for key in ops_by_model:
                    ops_by_model[key].pop(name, None)
                    ops_skipped[key].pop(name, None)
                    collected = (result.get('ops') or {}).get(key)
                    if collected:
                        ops_by_model[key][name] = collected
                        op_schemas[key].update((result.get('op_schemas') or {}).get(key) or {})
                    elif result.get('status') == 'ok':
                        ops_skipped[key][name] = ((result.get('ops_error') or {}).get(key)
                                                  or 'no ops collected')
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

    written = [output_path]
    if collect:
        written += [aten_yaml_path, aten_md_path, func_yaml_path, func_md_path]
        seen = {row.get('device') for row in rows.values() if row.get('device')}
        for backend in list(CORE_BACKENDS) + sorted(seen - set(CORE_BACKENDS)):
            written += list(core_paths(backend))
    print('Wrote ' + ', '.join(written), file=sys.stderr)


if __name__ == '__main__':
    main()
