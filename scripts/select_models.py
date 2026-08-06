#!/usr/bin/env python3
"""Pick a representative subset of the timm zoo to publish graphs and .pt2 archives for.

timm registers ~1300 variants; publishing a serialized graph for each would be hundreds of
megabytes of near-duplicate JSON, most of it the same handful of architectures at different
widths. This picks a subset from data the repo already generates -- the operator
cross-references (which operators each model actually uses, and in which call configurations,
in every dialect the report catalogues) and `models.md`
(family, weight size, resolution) -- plus `model-popularity.yaml` (HuggingFace Hub downloads,
from `make models.popularity`) so "representative" means something checkable rather than a
matter of taste. See `pt2_export_core.selection` for the two-phase algorithm itself; this
script only parses timm's report format into candidates and injects timm's own pretrained-
weight lookup.

Reading a *committed* report rather than exporting means this runs in seconds and its result
is reviewable in a diff before any of the expensive work happens.
"""
import argparse
import os
import sys

from export_report import parse_existing, parse_existing_ops
from exportlib import pretrained_info
from pt2_export_core.catalog import CORE_BACKENDS
from pt2_export_core.selection import build_candidates, load_popularity, read_overrides, render_manifest, select


def load_candidates(models_md, ops_aten, ops_core):
    """Parse the committed reports and join them into candidates via
    `pt2_export_core.selection.build_candidates`.

    `ops_aten` is the matrix of the graph this repo publishes, so it decides which models are
    candidates and what each costs. `ops_core` is one file per backend: the same models lowered,
    a different operator set worth covering but not a different artifact, so they are merged and
    contribute coverage only.
    """
    rows = parse_existing(models_md)
    if not rows:
        sys.exit(f'{models_md}: no model rows found -- run `make report` first')
    ops_by_model, _, _ = parse_existing_ops(ops_aten)
    if not ops_by_model:
        sys.exit(f'{ops_aten}: no operator matrix found -- run `make report` first')
    core_by_model = {}
    for path in ops_core:
        matrix, _, _ = parse_existing_ops(path)
        core_by_model.update(matrix)
    return build_candidates(rows, ops_by_model, core_by_model)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-md', default=os.path.join(repo_root, 'models.md'))
    parser.add_argument('--ops-aten', default=os.path.join(repo_root, 'ops-aten.yaml'),
                        help='cross-reference of the graph that actually gets published; decides '
                             'candidacy and per-model cost')
    parser.add_argument('--ops-core', nargs='+',
                        default=[os.path.join(repo_root, f'ops-core-{backend}.yaml')
                                 for backend in CORE_BACKENDS],
                        help='core ATen cross-references, one per backend; folded into coverage so a '
                             'model gets credit for the decomposition units it exercises too')
    parser.add_argument('--popularity', default=os.path.join(repo_root, 'model-popularity.yaml'),
                        help='HuggingFace Hub download counts from `make models.popularity`; '
                             'missing/absent entries just fall back to the size-based tie-break')
    parser.add_argument('--output', default=os.path.join(repo_root, 'models-selected.yaml'))
    parser.add_argument('--target', type=int, default=100,
                        help='how many models to select. See coverage-curve.yaml '
                             '(`make models.curve`) for what count buys what coverage before '
                             'picking a different one -- op-config coverage gain per 10 models '
                             'drops from ~4-6pp to ~2.3pp around here, while committed size '
                             'keeps climbing linearly regardless')
    parser.add_argument('--max-nodes', type=int, default=1500,
                        help='per-model ATen graph node cap -- the graph that actually gets '
                             'committed, at roughly 1.8KB per node, so this is the real size '
                             'control; raising it admits large models that contribute little the '
                             'smaller ones do not')
    parser.add_argument('--max-weight', type=float, default=150.0,
                        help='per-model weight cap in MB, for what has to be built and held in RAM')
    parser.add_argument('--release-max-weight', type=float, default=100.0,
                        help='per-model weight cap in MB for the release tier, which ships real '
                             'pretrained weights and so pays the size in every download')
    args = parser.parse_args()

    candidates = load_candidates(args.models_md, args.ops_aten, args.ops_core)
    include, exclude = read_overrides(args.output)
    popularity = load_popularity(args.popularity)

    selected = select(candidates, args.target, args.max_nodes, args.max_weight, include, set(exclude),
                       popularity)
    text, summary = render_manifest(selected, candidates, args.target, args.max_nodes, args.max_weight,
                                     args.release_max_weight, include, exclude, popularity,
                                     pretrained_info=pretrained_info)
    with open(args.output, 'w') as f:
        f.write(text)

    print(f'Wrote {args.output}: {summary["models"]} models, {summary["op_configs_covered"]} op configs, '
          f'{summary["families_covered"]} families, {summary["release_models"]} releasable',
          file=sys.stderr)


if __name__ == '__main__':
    main()
