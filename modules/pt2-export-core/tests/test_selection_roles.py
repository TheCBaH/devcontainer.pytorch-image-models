from pt2_export_core.selection import (
    load_roles, op_unit, render_role_candidates, role_candidates, select,
)


def _candidate(family, ops, nodes=10, weight_mb=1.0):
    return {'family': family, 'nodes': nodes, 'ops': set(ops), 'weight_mb': weight_mb,
            'resolution': '224x224'}


def test_select_populates_role_units_for_first_contributor_only():
    common = op_unit('aten.relu.default', {})
    rare = op_unit('aten.adaptive_avg_pool2d.default', {'output_size': [1, 1]})
    candidates = {
        'a': _candidate('fam_a', {common, rare}),
        'b': _candidate('fam_b', {common}),  # contributes nothing new once 'a' is taken
    }
    role_units = {}
    selected = select(candidates, target=2, max_nodes=1000, max_weight_mb=100.0,
                      include=[], exclude=set(), role_units=role_units)
    assert dict(selected) or selected  # sanity: something was selected
    assert role_units['a'] == sorted({common, rare})
    assert 'b' not in role_units  # contributed nothing new -- no entry


def test_role_candidates_filters_by_allowlist():
    common = op_unit('aten.relu.default', {})
    rare = op_unit('aten.adaptive_avg_pool2d.default', {'output_size': [1, 1]})
    role_units = {'a': sorted([common, rare]), 'b': [common]}
    filtered = role_candidates(role_units, allowlist={'aten.adaptive_avg_pool2d.default'})
    assert filtered == {'a': [rare]}


def test_render_role_candidates_is_valid_yaml_and_sorted(tmp_path):
    import yaml
    text = render_role_candidates({'b': ['unit2'], 'a': ['unit1']})
    doc = yaml.safe_load(text)
    assert doc == {'candidates': {'a': ['unit1'], 'b': ['unit2']}}
    assert list(doc['candidates']) == ['a', 'b']  # sorted by name


def test_load_roles_missing_file_returns_empty(tmp_path):
    assert load_roles(str(tmp_path / 'does-not-exist.yaml')) == {}


def test_load_roles_reads_hand_maintained_file(tmp_path):
    path = tmp_path / 'models-roles.yaml'
    path.write_text('models:\n  convit_tiny:\n    - adaptive_pool_1x1\n')
    assert load_roles(str(path)) == {'convit_tiny': ['adaptive_pool_1x1']}
