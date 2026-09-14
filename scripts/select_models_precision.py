#!/usr/bin/env python3
"""Pick a representative subset of the timm zoo to publish precision (cast/autocast) graphs
for, one dtype (fp16 or bf16) at a time.

Sibling of `select_models.py`, reusing the same two-phase algorithm
(`pt2_export_core.selection`) and the same coverage-over-committed-reports approach, just over
a different pair of dialects: `ops-aten-{label}.yaml` (`model.to(dtype=...)`, meta-only, the
whole ~1300-model catalog -- this is the dialect that actually gets built and committed, so it
decides candidacy and per-model cost) folded together with `ops-func-autocast-{label}.yaml`
(a `torch.autocast('cpu', dtype=...)` forward wrapper, real-CPU-only and therefore restricted
to the GFLOPs/weight-capped subset `report.precision.autocast` covers) as an extra coverage
matrix -- a model earns its place by what either policy demonstrates, while what it costs to
commit is always the cast graph.

Reading committed reports rather than exporting means this runs in seconds and its result is
reviewable in a diff before `make models.fp16`/`make models.bf16` builds anything.
"""
import argparse
import os
import sys

from export_report import parse_existing, parse_existing_ops
from exportlib import pretrained_info
from pt2_export_core.selection import (
    ROLE_OP_ALLOWLIST, build_candidates, load_popularity, read_overrides, render_manifest,
    render_role_candidates, role_candidates, select,
)

DTYPE_LABEL = {'float16': 'fp16', 'bfloat16': 'bf16'}


def load_candidates(models_md, ops_aten, ops_autocast):
    """Parse the committed reports and join them via `pt2_export_core.selection.build_candidates`.

    `ops_aten` (the cast dialect) decides which models are candidates and what each costs --
    it is the one that gets built into `models-{fp16,bf16}/cast/`. `ops_autocast` contributes
    coverage only: a model absent from it (outside the autocast policy's GFLOPs/weight caps)
    can still be a candidate on its cast coverage alone.
    """
    rows = parse_existing(models_md)
    if not rows:
        sys.exit(f'{models_md}: no model rows found -- run `make report` first')
    ops_by_model, _, _ = parse_existing_ops(ops_aten)
    if not ops_by_model:
        sys.exit(f'{ops_aten}: no operator matrix found -- run `make report.precision` first')
    autocast_by_model, _, _ = parse_existing_ops(ops_autocast)
    if not autocast_by_model:
        sys.exit(f'{ops_autocast}: no operator matrix found -- '
                 f'run `make report.precision.autocast` first')
    return build_candidates(rows, ops_by_model, autocast_by_model)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dtype', required=True, choices=['float16', 'bfloat16'])
    parser.add_argument('--models-md', default=os.path.join(repo_root, 'models.md'))
    parser.add_argument('--ops-aten', default=None,
                        help='cast cross-reference; decides candidacy and per-model cost '
                             '(default: <repo root>/ops-aten-<label>.yaml)')
    parser.add_argument('--ops-autocast', default=None,
                        help='autocast cross-reference; coverage only, restricted subset '
                             '(default: <repo root>/ops-func-autocast-<label>.yaml)')
    parser.add_argument('--popularity', default=os.path.join(repo_root, 'model-popularity.yaml'),
                        help='HuggingFace Hub download counts from `make models.popularity`; '
                             'missing/absent entries just fall back to the size-based tie-break')
    parser.add_argument('--output', default=None,
                        help='default: <repo root>/models-<label>.yaml')
    parser.add_argument('--target', type=int, default=100,
                        help='how many models to select')
    parser.add_argument('--max-nodes', type=int, default=1500,
                        help='per-model cast-graph node cap -- the graph that actually gets '
                             'committed under models-<label>/cast/')
    parser.add_argument('--max-weight', type=float, default=150.0,
                        help='per-model fp32 weight cap in MB, for what has to be built and '
                             'held in RAM under either policy')
    parser.add_argument('--write-role-candidates', action='store_true')
    parser.add_argument('--role-candidates-output', default=None,
                        help='default: <repo root>/models-<label>-role-candidates.yaml')
    parser.add_argument('--role-allowlist', nargs='+', default=sorted(ROLE_OP_ALLOWLIST),
                        metavar='OP')
    args = parser.parse_args()

    label = DTYPE_LABEL[args.dtype]
    ops_aten = args.ops_aten or os.path.join(repo_root, f'ops-aten-{label}.yaml')
    ops_autocast = args.ops_autocast or os.path.join(repo_root, f'ops-func-autocast-{label}.yaml')
    output = args.output or os.path.join(repo_root, f'models-{label}.yaml')
    role_candidates_output = (args.role_candidates_output
                              or os.path.join(repo_root, f'models-{label}-role-candidates.yaml'))

    candidates = load_candidates(args.models_md, ops_aten, ops_autocast)
    include, exclude = read_overrides(output)
    popularity = load_popularity(args.popularity)

    # release_max_weight_mb has no meaning here -- these manifests feed `models.fp16`/
    # `models.bf16` (graphs only), never `cmd_release` -- so it is passed equal to max_weight,
    # which makes render_manifest's `release` flag simply track the one cap that applies.
    role_units = {} if args.write_role_candidates else None
    selected = select(candidates, args.target, args.max_nodes, args.max_weight, include,
                       set(exclude), popularity, role_units=role_units)
    text, summary = render_manifest(
        selected, candidates, args.target, args.max_nodes, args.max_weight, args.max_weight,
        include, exclude, popularity, pretrained_info=pretrained_info,
        script='scripts/select_models_precision.py', make_target=f'make models.select.{label}')
    # render_manifest's `release` field/comment describes select_models.py's own manifest,
    # where it means "shipped as a .pt2 on a tagged release" -- not true here, models-fp16.yaml/
    # models-bf16.yaml have no release archive, only the cast/autocast graphs `make
    # models.fp16`/`make models.bf16` build. Reword rather than fork the renderer; a no-op if
    # this prose ever moves in selection.py.
    text = text.replace(
        '# Models published as PT2 graphs and release archives.',
        f'# Models published as {label} cast/autocast PT2 graphs (models-{label}/cast/, '
        f'models-{label}/autocast/).',
    )
    text = text.replace(
        "# release   shipped as a .pt2 on tagged releases: has fetchable pretrained weights, fits\n"
        "#           under the release weight cap, and has a classification head (num_classes > 0)\n"
        "#           -- a self-supervised backbone with no head has nothing for the release\n"
        "#           archive's expected-top5 contract to report.",
        "# release   inherited from select_models.py's shared renderer: has fetchable pretrained\n"
        "#           weights within the weight cap and a classification head. This manifest has\n"
        "#           no release archive of its own (see models-selected.yaml for that) -- the\n"
        "#           flag only says a real checkpoint could back this model's graph.",
    )
    with open(output, 'w') as f:
        f.write(text)

    print(f'Wrote {output}: {summary["models"]} models, {summary["op_configs_covered"]} '
          f'(policy, config) units, {summary["families_covered"]} families', file=sys.stderr)

    if args.write_role_candidates:
        candidates_doc = role_candidates(role_units, set(args.role_allowlist))
        with open(role_candidates_output, 'w') as f:
            f.write(render_role_candidates(candidates_doc, script='scripts/select_models_precision.py'))
        print(f'Wrote {role_candidates_output}: {len(candidates_doc)} model(s)', file=sys.stderr)


if __name__ == '__main__':
    main()
