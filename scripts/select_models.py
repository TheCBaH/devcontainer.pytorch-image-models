#!/usr/bin/env python3
"""Pick a representative subset of the timm zoo to publish graphs and .pt2 archives for.

timm registers ~1300 variants; publishing a serialized graph for each would be hundreds of
megabytes of near-duplicate JSON, most of it the same handful of architectures at different
widths. This picks a subset from data the repo already generates -- `ops.yaml` (which core
ATen operators each model actually uses) and `models.md` (family, weight size, resolution) --
so "representative" means something checkable rather than a matter of taste:

  phase 1  greedy set cover over the core ATen operator set, scored by operators-gained per
           graph node, so the cheapest carrier of a rare operator wins over a large model
           that merely repeats common ones
  phase 2  the smallest so-far-unrepresented architecture family, repeatedly, until the
           target count is reached

Reading a *committed* report rather than exporting means this runs in seconds and its result
is reviewable in a diff before any of the expensive work happens.
"""
import argparse
import os
import sys

import yaml

from export_report import parse_existing, parse_existing_ops


def load_candidates(models_md, ops_yaml):
    """Join the two committed reports into {name: {family, nodes, ops, weight_mb, resolution}}.

    A model needs a row in both to be a candidate: `models.md` alone cannot say which
    operators it uses, and a model missing from `ops.yaml` is one whose decomposition timed
    out, which is not something to build a published artifact on.
    """
    rows = parse_existing(models_md)
    if not rows:
        sys.exit(f'{models_md}: no model rows found -- run `make report` first')
    ops_by_model, _, _ = parse_existing_ops(ops_yaml)
    if not ops_by_model:
        sys.exit(f'{ops_yaml}: no operator matrix found -- run `make report` first')

    candidates = {}
    for name, cells in ops_by_model.items():
        row = rows.get(name)
        if not row or row.get('status') != 'ok':
            continue
        try:
            weight_mb = float(row['weight_str'])
        except (TypeError, ValueError):
            continue
        candidates[name] = {
            'family': row.get('family', 'unknown'),
            'nodes': sum(count for _, _, count in cells),
            'ops': {op for op, _, _ in cells},
            'weight_mb': weight_mb,
            'resolution': row.get('resolution', ''),
        }
    return candidates


def pretrained_info(name):
    """(tag, hf_hub_id) for a model's default pretrained weights, or (None, None).

    A model can carry a pretrained *tag* while having nowhere to fetch it from -- timm's
    `test_*` architectures are the obvious case -- so the presence of a real source, not of
    a tag, is what decides whether a variant can ship as a runnable .pt2.
    """
    from timm.models import get_pretrained_cfg
    try:
        cfg = get_pretrained_cfg(name)
    except Exception:
        return None, None
    if not (cfg.hf_hub_id or cfg.url or cfg.file):
        return None, None
    return cfg.tag or None, cfg.hf_hub_id or None


def select(candidates, target, max_nodes, max_weight_mb, include, exclude):
    """Run the two phases and return [(name, phase)] in selection order."""
    eligible = {
        name: c for name, c in candidates.items()
        if name not in exclude and c['nodes'] <= max_nodes and c['weight_mb'] <= max_weight_mb
    }

    selected, covered, families = [], set(), set()

    def take(name, phase):
        selected.append((name, phase))
        covered.update(candidates[name]['ops'])
        families.add(candidates[name]['family'])

    # `include` is honoured verbatim and ahead of everything else: it is the escape hatch for
    # an operator or architecture the caps would otherwise price out, and it deliberately
    # ignores max_nodes/max_weight_mb -- an explicit choice outranks a heuristic one.
    for name in include:
        if name in candidates and name not in exclude:
            take(name, 'include')
        else:
            print(f'warning: include list names {name!r}, which is not an exportable model',
                  file=sys.stderr)

    # Phase 1 -- gain per node, so a 51-node resnet10t that contributes 20 operators beats a
    # 1300-node transformer that contributes 25. Ties break on node count then name, which is
    # what makes the whole selection reproducible.
    while len(selected) < target:
        best, best_score = None, None
        for name, c in eligible.items():
            gain = len(c['ops'] - covered)
            if not gain:
                continue
            score = (-gain / c['nodes'], c['nodes'], name)
            if best_score is None or score < best_score:
                best, best_score = name, score
        if best is None:
            break
        take(best, 'ops')

    # Phase 2 -- breadth. Operator coverage saturates well before the target, and the
    # remaining budget is better spent on architecture families nothing represents yet than
    # on more variants of the ones already in.
    for name, c in sorted(eligible.items(), key=lambda kv: (kv[1]['nodes'], kv[1]['weight_mb'], kv[0])):
        if len(selected) >= target:
            break
        if c['family'] in families:
            continue
        take(name, 'family')

    return selected


def render(selected, candidates, args, include, exclude):
    lines = [
        '# Models published as PT2 graphs and release archives.',
        '#',
        '# Generated by scripts/select_models.py from ops.yaml + models.md -- regenerate with',
        '# `make models.select`. Hand edits are overwritten, except `include` and `exclude`,',
        '# which are read back and honoured verbatim.',
        '#',
        '# phase   why the model is here: `ops` = it carried operators nothing else cheaper did,',
        '#         `family` = its architecture family had no representative, `include` = named by hand.',
        '# nodes   core ATen nodes in its decomposed graph; the cost driver for the committed JSON.',
        '# release shipped as a .pt2 on tagged releases: has fetchable pretrained weights and fits',
        '#         under the release weight cap.',
        '',
    ]

    all_ops = set().union(*(c['ops'] for c in candidates.values()))
    all_families = {c['family'] for c in candidates.values()}
    covered_ops = set().union(*(candidates[n]['ops'] for n, _ in selected)) if selected else set()
    covered_families = {candidates[n]['family'] for n, _ in selected}
    missing = sorted(all_ops - covered_ops)

    summary = {
        'target': args.target,
        'max_nodes': args.max_nodes,
        'max_weight_mb': args.max_weight,
        'release_max_weight_mb': args.release_max_weight,
        'models': len(selected),
        'ops_covered': f'{len(covered_ops)}/{len(all_ops)}',
        'families_covered': f'{len(covered_families)}/{len(all_families)}',
        'total_nodes': sum(candidates[n]['nodes'] for n, _ in selected),
    }
    document = {'selection': summary}
    if missing:
        # Named rather than silently absent: an uncovered operator is a real gap in what the
        # published graphs demonstrate, and the only fix is an `include` entry, so the reader
        # needs to know it exists.
        document['uncovered_ops'] = missing
    document['include'] = list(include)
    document['exclude'] = list(exclude)

    models = {}
    for name, phase in sorted(selected):
        c = candidates[name]
        tag, hub_id = pretrained_info(name)
        entry = {
            'family': c['family'],
            'phase': phase,
            'nodes': c['nodes'],
            'weight_mb': c['weight_mb'],
            'resolution': c['resolution'],
            # The one place the release tier is defined. export_pt2.release_names reads this
            # flag rather than re-deriving it, so what ships on a tag is decided once.
            'release': bool(hub_id) and c['weight_mb'] <= args.release_max_weight,
        }
        if tag:
            entry['pretrained_tag'] = tag
        if hub_id:
            entry['hf_hub_id'] = hub_id
        models[name] = entry
    document['models'] = models
    summary['release_models'] = sum(1 for e in models.values() if e['release'])

    lines.append(yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=100))
    return '\n'.join(lines), summary


def read_overrides(path):
    """Carry `include`/`exclude` across a regeneration -- they are the file's only hand-written
    part, and losing them on every `make models.select` would make them useless."""
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    return list(document.get('include') or []), list(document.get('exclude') or [])


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-md', default=os.path.join(repo_root, 'models.md'))
    parser.add_argument('--ops', default=os.path.join(repo_root, 'ops.yaml'))
    parser.add_argument('--output', default=os.path.join(repo_root, 'models-selected.yaml'))
    parser.add_argument('--target', type=int, default=60, help='how many models to select')
    parser.add_argument('--max-nodes', type=int, default=2500,
                        help='per-model graph node cap. The committed JSON costs ~4KB per node, so '
                             'this is the real size control; raising it admits large models that '
                             'contribute little the smaller ones do not')
    parser.add_argument('--max-weight', type=float, default=150.0,
                        help='per-model weight cap in MB, for what has to be built and held in RAM')
    parser.add_argument('--release-max-weight', type=float, default=100.0,
                        help='per-model weight cap in MB for the release tier, which ships real '
                             'pretrained weights and so pays the size in every download')
    args = parser.parse_args()

    candidates = load_candidates(args.models_md, args.ops)
    include, exclude = read_overrides(args.output)

    selected = select(candidates, args.target, args.max_nodes, args.max_weight, include, set(exclude))
    text, summary = render(selected, candidates, args, include, exclude)
    with open(args.output, 'w') as f:
        f.write(text)

    print(f'Wrote {args.output}: {summary["models"]} models, {summary["ops_covered"]} operators, '
          f'{summary["families_covered"]} families, {summary["release_models"]} releasable',
          file=sys.stderr)


if __name__ == '__main__':
    main()
