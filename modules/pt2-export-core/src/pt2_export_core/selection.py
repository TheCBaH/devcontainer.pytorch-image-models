"""Coverage-based selection of a representative subset of a model zoo to publish, plus the
coverage-vs-count curve used to pick a target size.

A model registers many more variants than are worth publishing a full graph for; this picks
a subset from data a zoo's own report already generated -- an ops-by-model matrix (which
core ATen operators each model uses, and in which call configurations) and per-model rows
(family, weight size, ...) -- plus optional per-repo download counts, so "representative"
means something checkable rather than a matter of taste:

  phase 1  greedy set cover over (operator, call configuration) pairs -- not just operator
           names, since most operators carry several recorded configurations (dtype, rank,
           kwargs) and a graph exercising only the commonest one demonstrates less than one
           that also hits its edges -- scored by units gained per graph node, so the
           cheapest carrier of a rare unit wins over a large model that merely repeats
           common ones. Ties (several models fit the same still-uncovered unit equally well)
           break on download count, then node count, then name.
  phase 2  the cheapest so-far-unrepresented architecture family, repeatedly, until the
           target count is reached. Cost orders which family goes next -- popularity plays no
           part in that, coverage breadth is the point of this phase -- but when a family has
           several eligible variants, i.e. several models could fill that one slot, the most-
           downloaded of them becomes its representative.

This module has no opinion about where `rows`/`ops_by_model` came from -- that is a zoo's own
report format, parsed by the caller -- nor about pretrained-weight lookup, which callers
inject as a `pretrained_info(name) -> (tag, hf_hub_id)` callback so this stays usable by any
zoo, not just the one whose report format populated `rows`.
"""
import json
import os
import sys

import yaml


def op_unit(op, config):
    """A hashable, readable unit of coverage: one operator in one recorded call configuration.

    Most operators carry several configurations -- different dtypes, ranks, kwargs -- so
    treating the operator name alone as "covered" once any variant of it appears would
    under-count what the published graphs actually exercise.
    """
    return f'{op} {json.dumps(config, sort_keys=True)}'


def build_candidates(rows, ops_by_model):
    """Join parsed report rows with an ops-by-model matrix into
    {name: {family, nodes, ops, weight_mb, resolution}}.

    A model needs an entry in both to be a candidate: `rows` alone cannot say which
    operators it uses, and a model missing from `ops_by_model` is one whose decomposition
    timed out, which is not something to build a published artifact on. `ops` here is a set
    of `op_unit()` results, i.e. (operator, configuration) pairs, not bare operator names.
    """
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
            'ops': {op_unit(op, config) for op, config, _ in cells},
            'weight_mb': weight_mb,
            'resolution': row.get('resolution', ''),
        }
    return candidates


def load_popularity(path):
    """{model_name: downloads}, summed across every pretrained tag that name ships.

    The popularity file is keyed by repo name relative to the source org
    (`efficientnet_b0.ra_in1k`), one repo per tag; a model with several tags -- and therefore
    several download counts -- should still rank as one popularity figure, so this collapses
    each repo name to the model name before the first `.` and sums. Missing the file entirely
    (never fetched) degrades to "no popularity data", not an error: phase 1 falls back to its
    node/name tie-break and phase 2 falls back to smallest-family-first, which is exactly the
    old behaviour.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    popularity = {}
    for repo_name, downloads in (document.get('downloads') or {}).items():
        name = repo_name.split('.')[0]
        popularity[name] = popularity.get(name, 0) + downloads
    return popularity


def select(candidates, target, max_nodes, max_weight_mb, include, exclude, popularity=None):
    """Run the two phases and return [(name, phase)] in selection order."""
    popularity = popularity or {}
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

    # Phase 1 -- gain per node, so a cheap model that contributes many (op, config) units
    # beats an expensive one that contributes only slightly more. Ties -- several models
    # would gain the same units at the same cost, i.e. several models fit this coverage need
    # equally -- break on downloads first (the popular one wins), then node count, then name
    # for anything nobody downloads.
    while len(selected) < target:
        best, best_score = None, None
        for name, c in eligible.items():
            gain = len(c['ops'] - covered)
            if not gain:
                continue
            score = (-gain / c['nodes'], -popularity.get(name, 0), c['nodes'], name)
            if best_score is None or score < best_score:
                best, best_score = name, score
        if best is None:
            break
        take(best, 'ops')

    # Phase 2 -- breadth. Operator coverage saturates well before the target, and the
    # remaining budget is better spent on architecture families nothing represents yet than
    # on more variants of the ones already in. Which family goes next is still cost-driven --
    # the cheapest so-far-unrepresented family, so the remaining budget stretches over as many
    # families as possible -- popularity has no say in that ordering. It only decides which
    # variant represents a family once that family is up: if several eligible variants could
    # fill the slot, the most-downloaded one does.
    by_family = {}
    for name, c in eligible.items():
        by_family.setdefault(c['family'], []).append(name)

    def cheapest(names):
        return min((eligible[n]['nodes'], eligible[n]['weight_mb'], n) for n in names)

    for family, names in sorted(by_family.items(), key=lambda kv: cheapest(kv[1])):
        if len(selected) >= target:
            break
        if family in families:
            continue
        representative = min(names, key=lambda n: (-popularity.get(n, 0), eligible[n]['nodes'], n))
        take(representative, 'family')

    return selected


def render_manifest(selected, candidates, target, max_nodes, max_weight_mb, release_max_weight_mb,
                     include, exclude, popularity=None, pretrained_info=None,
                     script='scripts/select_models.py', make_target='make models.select'):
    """Render the selection manifest. `pretrained_info(name) -> (tag, hf_hub_id)` is injected
    by the caller (rather than imported here) so this stays usable by any zoo's own registry,
    not just the one whose report format populated `candidates`."""
    popularity = popularity or {}
    lines = [
        '# Models published as PT2 graphs and release archives.',
        '#',
        f'# Generated by {script} from ops.yaml + models.md + model-popularity.yaml',
        f'# -- regenerate with `{make_target}`. Hand edits are overwritten, except `include`',
        '# and `exclude`, which are read back and honoured verbatim.',
        '#',
        '# phase     why the model is here: `ops` = it carried (operator, configuration) units nothing',
        '#           else cheaper did, `family` = its architecture family had no representative,',
        '#           `include` = named by hand.',
        '# nodes     core ATen nodes in its decomposed graph; the cost driver for the committed JSON.',
        '# downloads HuggingFace Hub downloads (last 30 days, summed across pretrained tags); decides',
        '#           which model fills a coverage slot when several could, not which slot goes next.',
        '#           Absent if the model has no Hub weights or fetch_popularity.py has not seen it.',
        '# release   shipped as a .pt2 on tagged releases: has fetchable pretrained weights and fits',
        '#           under the release weight cap.',
        '',
    ]

    all_ops = set().union(*(c['ops'] for c in candidates.values()))
    all_families = {c['family'] for c in candidates.values()}
    covered_ops = set().union(*(candidates[n]['ops'] for n, _ in selected)) if selected else set()
    covered_families = {candidates[n]['family'] for n, _ in selected}
    missing = sorted(all_ops - covered_ops)

    summary = {
        'target': target,
        'max_nodes': max_nodes,
        'max_weight_mb': max_weight_mb,
        'release_max_weight_mb': release_max_weight_mb,
        'models': len(selected),
        'op_configs_covered': f'{len(covered_ops)}/{len(all_ops)}',
        'families_covered': f'{len(covered_families)}/{len(all_families)}',
        'total_nodes': sum(candidates[n]['nodes'] for n, _ in selected),
    }
    document = {'selection': summary}
    if missing:
        # Named rather than silently absent: an uncovered (operator, configuration) unit is a
        # real gap in what the published graphs demonstrate, and the only fix is an `include`
        # entry, so the reader needs to know it exists.
        document['uncovered_op_configs'] = missing
    document['include'] = list(include)
    document['exclude'] = list(exclude)

    models = {}
    for name, phase in sorted(selected):
        c = candidates[name]
        tag, hub_id = pretrained_info(name) if pretrained_info else (None, None)
        entry = {
            'family': c['family'],
            'phase': phase,
            'nodes': c['nodes'],
            'weight_mb': c['weight_mb'],
            'resolution': c['resolution'],
            # The one place the release tier is defined. A driver's own archive builder reads
            # this flag rather than re-deriving it, so what ships on a tag is decided once.
            'release': bool(hub_id) and c['weight_mb'] <= release_max_weight_mb,
        }
        if tag:
            entry['pretrained_tag'] = tag
        if hub_id:
            entry['hf_hub_id'] = hub_id
        if popularity.get(name):
            entry['downloads'] = popularity[name]
        models[name] = entry
    document['models'] = models
    summary['release_models'] = sum(1 for e in models.values() if e['release'])

    lines.append(yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=100))
    return '\n'.join(lines), summary


def read_overrides(path):
    """Carry `include`/`exclude` across a regeneration -- they are the file's only hand-written
    part, and losing them on every regeneration would make them useless."""
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    return list(document.get('include') or []), list(document.get('exclude') or [])


def curve(candidates, popularity, max_nodes, max_weight_mb, step):
    """Report (operator, configuration) coverage and family breadth as a function of model
    count, in steps of `step`, from `step` up to the point the algorithm saturates on its own
    (no eligible candidate gains anything further) -- so the actual shape of the
    coverage-vs-count curve is something to look at and decide a target from, rather than
    something to guess.
    """
    all_ops = set().union(*(c['ops'] for c in candidates.values()))
    all_families = {c['family'] for c in candidates.values()}

    # Uncapped run to find where the algorithm stops adding anything on its own -- the natural
    # ceiling, past which a bigger target would just repeat this same selection.
    saturated = select(candidates, 10**9, max_nodes, max_weight_mb, [], set(), popularity)
    max_models = len(saturated)

    targets = list(range(step, max_models, step)) + [max_models]
    points = []
    for target in targets:
        selected = select(candidates, target, max_nodes, max_weight_mb, [], set(), popularity)
        covered_ops = set().union(*(candidates[n]['ops'] for n, _ in selected)) if selected else set()
        covered_families = {candidates[n]['family'] for n, _ in selected}
        points.append({
            'target': target,
            'models': len(selected),
            'op_configs_covered': len(covered_ops),
            'op_configs_total': len(all_ops),
            'op_configs_pct': round(100 * len(covered_ops) / len(all_ops), 1),
            'families_covered': len(covered_families),
            'families_total': len(all_families),
            'families_pct': round(100 * len(covered_families) / len(all_families), 1),
            'total_nodes': sum(candidates[n]['nodes'] for n, _ in selected),
        })
    return points, max_models, len(all_ops), len(all_families)
