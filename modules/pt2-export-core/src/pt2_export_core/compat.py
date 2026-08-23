"""Graph-vs-runnable compatibility classification (suggestion #3).

Four states, not three, so a smoke-scope report can never be mistaken for full release
evidence:
  runnable      release-tier, attempted in this build, archive produced, every image passed.
  graph_only    not release-tier (never has an archive, regardless of scope).
  unsupported   release-tier, attempted, interpreter failed. Only ever appears in a
                scope: smoke report -- a scope: full release hard-fails before publishing if
                this happens (see the pack-results lifecycle in cmd_release).
  unverified    release-tier, but not attempted in this particular build (outside a
                scope: smoke run's --only set).
"""
import hashlib
import os

from . import opgraph

NOT_RELEASE_TIER = {'result': 'not_attempted', 'reason': 'not_release_tier'}
OUTSIDE_SMOKE_SCOPE = {'result': 'not_attempted', 'reason': 'outside_smoke_scope'}
PRIOR_BACKEND_FAILED = {'result': 'not_attempted', 'reason': 'prior_backend_failed'}


def static_graph_facts(models_dir, name):
    """{'histogram', 'config_variants', 'dtypes'} from a committed op_facts.json.

    Checks op_facts.json's model_json_sha256 against the actually-committed model.json
    before trusting anything in it -- a mismatch (possible only if something edited one file
    without the other outside this pipeline) is a named hard failure, since every fact below
    would otherwise silently describe the wrong graph.
    """
    model_dir = os.path.join(models_dir, name, 'models')
    model_json_path = os.path.join(model_dir, 'model.json')
    op_facts_path = os.path.join(model_dir, 'op_facts.json')
    if not os.path.exists(op_facts_path):
        raise FileNotFoundError(f'{name}: no op_facts.json at {op_facts_path} -- run `make models` first')

    with open(op_facts_path, 'rb') as f:
        op_facts = opgraph.strict_json_loads(f.read())

    with open(model_json_path, 'rb') as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    if actual_sha256 != op_facts['model_json_sha256']:
        raise ValueError(
            f'{name}: op_facts.json.model_json_sha256 ({op_facts["model_json_sha256"]}) does '
            f'not match the committed model.json ({actual_sha256}) -- refusing to trust static '
            'facts that may describe a different graph than the one actually committed')

    histogram = {}
    dtypes = set()
    for target, config, count in op_facts['ops']:
        histogram[target] = histogram.get(target, 0) + count
        dtype = config.get('out_dtype')
        if dtype:
            dtypes.add(dtype)

    return {
        'histogram': histogram,
        'config_variants': len(op_facts['ops']),
        'dtypes': sorted(dtypes),
    }


def classify(*, is_release_tier, attempted, pack_result=None):
    """(classification, {'interpreter': ..., 'aot_inductor_cpu': ...}) for one model in one
    particular build. `pack_result` is this model's own entry from pack-results.json (the
    full per-name result cmd_release now collects, not just a bare error string), None if
    `attempted` is False.
    """
    if not is_release_tier:
        return 'graph_only', {'interpreter': dict(NOT_RELEASE_TIER),
                              'aot_inductor_cpu': dict(NOT_RELEASE_TIER)}
    if not attempted:
        return 'unverified', {'interpreter': dict(OUTSIDE_SMOKE_SCOPE),
                              'aot_inductor_cpu': dict(OUTSIDE_SMOKE_SCOPE)}

    if pack_result.get('status') == 'ok':
        aoti = pack_result.get('aoti')
        if aoti is None:
            aoti_backend = dict(PRIOR_BACKEND_FAILED)  # AOTI attempt did not run at all
        elif aoti.get('status') == 'ok':
            aoti_backend = {'result': 'passed', 'environment': aoti.get('environment', {})}
            if aoti.get('hint'):
                aoti_backend['hint'] = aoti['hint']
        else:
            aoti_backend = {'result': 'failed', 'error': (aoti.get('error') or '')[:300],
                            'environment': aoti.get('environment', {})}
            if aoti.get('hint'):
                aoti_backend['hint'] = aoti['hint']
        return 'runnable', {'interpreter': {'result': 'passed'}, 'aot_inductor_cpu': aoti_backend}

    interpreter = {'result': 'failed', 'error': (pack_result.get('error') or '')[:300]}
    return 'unsupported', {'interpreter': interpreter, 'aot_inductor_cpu': dict(PRIOR_BACKEND_FAILED)}


def render_compat_report(*, scope, selected_names, release_names, attempted_names,
                         models_dir, pack_results):
    """`pack_results`: {name: pack-results.json entry}, only for names actually attempted in
    this build. `attempted_names` is the exact --only subset for scope: smoke, or the full
    release tier for scope: full.
    """
    if scope not in ('full', 'smoke'):
        raise ValueError(f'unknown scope: {scope!r}')

    models = {}
    for name in sorted(selected_names):
        is_release_tier = name in release_names
        attempted = name in attempted_names
        pack_result = pack_results.get(name) if attempted else None
        classification, backends = classify(is_release_tier=is_release_tier, attempted=attempted,
                                            pack_result=pack_result)
        graph = static_graph_facts(models_dir, name)
        models[name] = {'classification': classification, 'graph': graph, 'backends': backends}

    document = {
        'schema_version': 1,
        'scope': scope,
        'selected_models': {'count': len(selected_names), 'names': sorted(selected_names)},
        'release_models': {'count': len(release_names), 'names': sorted(release_names)},
        'backends': ['interpreter', 'aot_inductor_cpu'],
        'models': models,
    }
    if scope == 'smoke':
        document['attempted_models'] = {'count': len(attempted_names), 'names': sorted(attempted_names)}
    return document
