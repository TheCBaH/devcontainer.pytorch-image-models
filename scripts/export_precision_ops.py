#!/usr/bin/env python3
"""Report the ATen operator cross-reference for every timm model cast, or autocast, to
float16/bfloat16.

Sibling of `export_report.py`'s reports, generated the same way (subprocess-isolated,
`torch.export.export`), under one of two policies -- which do not share a dialect:

- `--policy cast` (default): `model.to(dtype=...)` plus a matching input -- what direct
  parameter/buffer casting looks like operator-by-operator, read from the same undecomposed
  ATen graph `ops-aten.md` documents (`ops-aten-fp16.yaml`/`.md` etc.). A plain dtype-metadata
  change, so it traces correctly on the **meta** device (no real weights needed) and covers the
  whole ~1300-model catalog affordably, with a real-CPU retry only for the minority whose
  *construction* (not the cast) needs real data -- same policy export_report.py's own
  worker_main uses for the fp32 baseline.
- `--policy autocast`: a `torch.autocast('cpu', dtype=...)` forward wrapper around an
  otherwise-fp32 model and input, read from the *functional* ATen graph `ops-func.md`
  documents instead (`ops-func-autocast-fp16.yaml`/`.md` etc.) -- the raw ATen graph captures
  autocast as one opaque `wrap_with_autocast` higher-order node with no per-operator detail at
  all, and only functionalizing (an empty decomp table) unwraps it. Autocast's dispatch also
  never actually engages on the meta backend -- a meta-traced autocast graph comes back
  uniformly fp32, indistinguishable from "had no effect" -- so this policy always runs on
  **real CPU tensors**, and is therefore capped by GFLOPs and weight size (read from the
  committed `models.md`) rather than run over the whole catalog: see `precision.md` for why a
  compute-cheap-but-wide model can still be expensive to save/run in parallel.

Every model is evaluated in its own isolated subprocess, exactly like export_report.py.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time

from exportlib import resolved_input_size
from export_report import parse_existing
from pt2_export_core.catalog import Dialect, render_ops_md, render_ops_yaml
from pt2_export_core.harness import cpu_count, globs, run_worker as _run_worker
from pt2_export_core.opgraph import collect_ops

DTYPE_LABEL = {'float16': 'fp16', 'bfloat16': 'bf16'}

_CAST_PROSE = (
    'Operators are read from the graph `torch.export.export()` hands back -- the same ATen '
    'dialect as [`ops-aten.md`](ops-aten.md), with `conv2d`, `linear`, `layer_norm` and '
    '`scaled_dot_product_attention` still whole -- after `model.to(dtype=torch.{dtype_name})` and a '
    'matching example input, rather than fp32. This is what direct parameter/buffer casting '
    'looks like operator-by-operator: `out_dtype` in the configurations below reads `{label}` '
    'wherever the graph actually carries it post-cast, and still `f32`/`i64`/`bool` wherever a '
    'value legitimately does not follow the cast (positional/index math, masks). Autocast is a '
    'different policy with a different cost profile -- see '
    '[`ops-func-autocast-{label}.md`](ops-func-autocast-{label}.md).'
)

_AUTOCAST_PROSE = (
    'Operators are read from the graph `run_decompositions(decomp_table={{}})` produces -- the '
    'same functional ATen dialect as [`ops-func.md`](ops-func.md), *not* the raw '
    '[`ops-aten.md`](ops-aten.md)/[`ops-aten-{label}.md`](ops-aten-{label}.md) dialect -- for a '
    '`torch.autocast(\'cpu\', dtype=torch.{dtype_name})` forward wrapper around an otherwise-fp32 '
    'model and input, rather than `model.to(dtype=...)`. The raw ATen graph `torch.export.export()` '
    'hands back for an autocast-wrapped model is a single opaque `wrap_with_autocast` higher-order '
    'node with no `_schema` (autocast is captured as an uninlined subgraph, not as ordinary '
    'operator calls) -- functionalizing is what unwraps it into the individual operators autocast '
    'actually dispatched, which is why this dialect, not the plain ATen one, is what this cross-'
    'reference needs. Autocast chooses precision per eligible operation rather than casting every '
    'parameter, so `out_dtype` in the configurations below is genuinely mixed: `{label}` wherever '
    'autocast\'s own rules picked low precision for that op, `f32` wherever it kept an operation '
    '(or an input/output boundary) in full precision, plus `i64`/`bool` for positional/index math '
    'and masks, which were never candidates either way. Autocast\'s dispatch never actually engages '
    'on the meta backend used for direct casting -- a meta-traced autocast graph comes back '
    'uniformly fp32, indistinguishable from "had no effect" -- so this cross-reference runs on '
    'real CPU tensors and is restricted to models at or under {gflops_cap:g} GFLOPs and '
    '{weight_cap:g}MB of fp32 weight (both from `models.md`): a compute-cheap-but-wide model can '
    'still be expensive to save/run in parallel, the same reason '
    '[`precision.md`](precision.md)\'s own zoo-wide sweep caps the model set it verifies '
    'numerically.'
)


def cast_dialect(dtype_name):
    label = DTYPE_LABEL[dtype_name]
    return Dialect(f'aten-{label}', f'ATen (cast to {dtype_name})', None,
                    f'ATen (torch.export.export, model+input cast to {dtype_name} before export)',
                    _CAST_PROSE.format(dtype_name=dtype_name, label=label))


def autocast_dialect(dtype_name, gflops_cap, weight_cap):
    label = DTYPE_LABEL[dtype_name]
    return Dialect(f'func-autocast-{label}', f'functional ATen (autocast to {dtype_name})', None,
                    f"functional ATen (torch.export.export + run_decompositions(decomp_table={{}}), "
                    f"torch.autocast('cpu', dtype={dtype_name}) forward wrapper)",
                    _AUTOCAST_PROSE.format(dtype_name=dtype_name, label=label,
                                            gflops_cap=gflops_cap, weight_cap=weight_cap))


def run_worker(model_name, policy, dtype_name, max_res, timeout):
    argv = ['--worker', model_name, '--policy', policy, '--dtype', dtype_name, '--max-res', max_res]
    return _run_worker(__file__, argv, model_name, timeout)


def _autocast_model_cls(torch):
    class _AutocastModel(torch.nn.Module):
        def __init__(self, model, dtype):
            super().__init__()
            self.model = model
            self.dtype = dtype

        def forward(self, x):
            with torch.autocast('cpu', dtype=self.dtype):
                return self.model(x)
    return _AutocastModel


def worker_main(model_name, policy, dtype_name, max_res):
    import torch
    import timm
    from timm.models import model_entrypoint

    dtype = getattr(torch, dtype_name)
    result = {'name': model_name, 'status': None, 'error': None}
    try:
        result['family'] = model_entrypoint(model_name).__module__.rsplit('.', 1)[-1]
    except Exception:
        result['family'] = 'unknown'

    def try_cast_export(device):
        if device == 'meta':
            with torch.device('meta'):
                model = timm.create_model(model_name, pretrained=False)
        else:
            model = timm.create_model(model_name, pretrained=False)
        model = model.eval().to(dtype=dtype)
        input_size = resolved_input_size(model.default_cfg, max_res)
        example = torch.empty(1, *input_size, dtype=dtype, device=device)
        with torch.no_grad():
            return torch.export.export(model, (example,))

    def try_autocast_export():
        # Real CPU tensors throughout: autocast's dispatch only does anything on a real
        # backend. Fresh random weights per call, never a warmed/reused model -- see the
        # precision investigation's ConViT `rel_indices` reproducibility trap for why a model
        # cast/wrapped after its first real forward can silently keep stale fp32 state.
        torch.manual_seed(123)
        base = timm.create_model(model_name, pretrained=False).eval()
        model = _autocast_model_cls(torch)(base, dtype)
        input_size = resolved_input_size(base.default_cfg, max_res)
        example = torch.randn(1, *input_size)
        with torch.no_grad():
            ep = torch.export.export(model, (example,))
            # Autocast is captured as a single opaque wrap_with_autocast higher-order node with
            # no _schema -- collect_ops (which only looks at the top-level graph) would see
            # nothing at all without this. An empty decomp table costs only functionalization,
            # the same dialect ops-func.md already documents.
            return ep.run_decompositions(decomp_table={})

    if policy == 'cast':
        try:
            ep = try_cast_export('meta')
            result['status'] = 'ok'
        except Exception:
            # Same rationale as export_report.py's worker_main: a minority of architectures
            # call real-data ops from config/init logic rather than forward, which meta
            # tensors can't satisfy -- a false negative for the cast itself, so retry for
            # real before giving up.
            try:
                ep = try_cast_export('cpu')
                result['status'] = 'ok'
            except Exception as e2:
                result['status'] = 'export_failed'
                result['error'] = str(e2)[:300]
                print(json.dumps(result))
                return
    else:
        try:
            ep = try_autocast_export()
            result['status'] = 'ok'
        except Exception as e:
            result['status'] = 'export_failed'
            result['error'] = str(e)[:300]
            print(json.dumps(result))
            return

    try:
        ops, schemas = collect_ops(ep)
        result['ops'], result['op_schemas'] = ops, schemas
    except Exception as e:
        result['status'] = 'ops_failed'
        result['error'] = str(e)[:300]
    print(json.dumps(result))


def autocast_candidates(models_md, gflops_cap, weight_cap_mb):
    """Models eligible for the real-CPU autocast policy: exported ok in the fp32 report, at or
    under both caps. Mirrors the precision investigation's own Stage B filter (see
    `precision.md`) -- weight matters as much as GFLOPs, since a compute-cheap-but-wide model
    still costs a large `.pt2` save per parallel worker."""
    rows = parse_existing(models_md)
    names = []
    for name, row in rows.items():
        if row.get('status') != 'ok' or row.get('gflops') is None:
            continue
        try:
            weight_mb = float(row.get('weight_str') or '')
        except ValueError:
            continue
        if row['gflops'] <= gflops_cap and weight_mb <= weight_cap_mb:
            names.append(name)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', required=True, choices=['float16', 'bfloat16'])
    parser.add_argument('--policy', default='cast', choices=['cast', 'autocast'])
    parser.add_argument('--gflops-cap', type=float, default=50.0,
                         help='autocast only: skip models above this fp32 GFLOPs figure')
    parser.add_argument('--weight-cap-mb', type=float, default=150.0,
                         help='autocast only: skip models above this fp32 weight size')
    parser.add_argument('--filter', default='', help='fnmatch glob(s), comma-separated')
    parser.add_argument('--exclude', default='', help='fnmatch glob(s), comma-separated')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=cpu_count())
    parser.add_argument('--timeout', type=float, default=None,
                         help='per-model subprocess timeout; default 120s for cast (meta, '
                              'cheap), 180s for autocast (real CPU compute)')
    parser.add_argument('--max-res', type=int, default=224)
    parser.add_argument('--models-md', default=None, help='autocast only: source of the '
                         'GFLOPs/weight caps, default <repo root>/models.md')
    parser.add_argument('--yaml-output', default=None)
    parser.add_argument('--md-output', default=None)
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker, args.policy, args.dtype, args.max_res)
        return

    import timm
    import torch

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timeout = args.timeout or (180.0 if args.policy == 'autocast' else 120.0)

    if args.policy == 'autocast':
        dialect = autocast_dialect(args.dtype, args.gflops_cap, args.weight_cap_mb)
        models_md = args.models_md or os.path.join(repo_root, 'models.md')
        names = autocast_candidates(models_md, args.gflops_cap, args.weight_cap_mb)
        if args.filter or args.exclude:
            import fnmatch
            include = globs(args.filter) if args.filter else None
            exclude = globs(args.exclude) if args.exclude else []
            if isinstance(include, str):
                include = [include]
            if isinstance(exclude, str):
                exclude = [exclude]
            names = [n for n in names
                     if (not include or any(fnmatch.fnmatch(n, p) for p in include))
                     and not any(fnmatch.fnmatch(n, p) for p in exclude)]
    else:
        dialect = cast_dialect(args.dtype)
        names = timm.list_models(filter=globs(args.filter), exclude_filters=globs(args.exclude),
                                  pretrained=False)

    yaml_path = args.yaml_output or os.path.join(repo_root, dialect.yaml_name)
    md_path = args.md_output or os.path.join(repo_root, dialect.md_name)

    if args.limit:
        names = names[:args.limit]
    total = len(names)
    print(f'Running {total} models {args.policy} to {args.dtype} with {args.workers} workers '
          f'(timeout={timeout}s each)...', file=sys.stderr)

    ops_by_model, op_schemas, skipped, families = {}, {}, {}, {}
    completed = 0
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_worker, name, args.policy, args.dtype, args.max_res, timeout): name
                   for name in names}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {'name': name, 'status': 'crashed', 'error': str(e)[:300]}
            families[name] = result.get('family', 'unknown')
            if result.get('status') == 'ok' and result.get('ops') is not None:
                ops_by_model[name] = result['ops']
                op_schemas.update(result.get('op_schemas') or {})
            else:
                skipped[name] = result.get('error') or result.get('status') or 'unknown failure'
            completed += 1
            print(f'[{completed}/{total}] {name}: {result.get("status")} '
                  f'({time.monotonic() - start:.0f}s elapsed)', file=sys.stderr)

    with open(yaml_path, 'w') as f:
        f.write(render_ops_yaml(ops_by_model, op_schemas, skipped, dialect, 'timm', timm.__version__,
                                 torch.__version__, script='scripts/export_precision_ops.py'))
    with open(md_path, 'w') as f:
        f.write(render_ops_md(ops_by_model, op_schemas, families, skipped, dialect, 'timm',
                               timm.__version__, torch.__version__, script='scripts/export_precision_ops.py'))
    print(f'Wrote {yaml_path}, {md_path} ({len(ops_by_model)}/{total} models ok)', file=sys.stderr)


if __name__ == '__main__':
    main()
