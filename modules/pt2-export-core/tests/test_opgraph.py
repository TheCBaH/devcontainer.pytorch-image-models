import json

import pytest

from pt2_export_core.opgraph import _plain, canonical_config, strict_json_loads


def test_plain_tags_nonfinite_floats_distinctly_from_strings():
    """A non-finite float and a string schema argument with the same text must not collide
    under the same canonical config key -- that would silently merge two distinct operator
    configurations into one."""
    neg_inf_key = canonical_config({'value': _plain(float('-inf'))})
    string_key = canonical_config({'value': _plain('-inf')})
    assert neg_inf_key != string_key
    assert _plain(float('-inf')) == {'$nonfinite_float': '-inf'}
    assert _plain(float('inf')) == {'$nonfinite_float': '+inf'}
    assert _plain(float('nan')) == {'$nonfinite_float': 'nan'}
    assert _plain('-inf') == '-inf'


def test_plain_leaves_finite_values_untouched():
    assert _plain(1) == 1
    assert _plain(1.5) == 1.5
    assert _plain(True) is True
    assert _plain(None) is None
    assert _plain([1, float('-inf'), 'x']) == [1, {'$nonfinite_float': '-inf'}, 'x']


def test_canonical_config_output_is_valid_json_for_nonfinite_values():
    key = canonical_config({'value': _plain(float('-inf'))})
    # json.loads (not strict_json_loads) must round-trip cleanly: no bare Infinity/NaN token.
    assert json.loads(key) == {'value': {'$nonfinite_float': '-inf'}}
    assert 'Infinity' not in key
    assert 'NaN' not in key


def test_canonical_config_rejects_bare_nonfinite_as_defense_in_depth():
    with pytest.raises(ValueError):
        canonical_config({'value': float('-inf')})


@pytest.mark.parametrize('document', [
    '{"a": NaN}',
    '{"a": Infinity}',
    '{"a": -Infinity}',
    '{"a": 1, "a": 2}',
    '{"a": {"b": 1, "b": 2}}',
])
def test_strict_json_loads_rejects_nonfinite_and_duplicate_keys(document):
    with pytest.raises(ValueError):
        strict_json_loads(document)


def test_strict_json_loads_accepts_well_formed_documents():
    assert strict_json_loads('{"a": 1, "b": [1, 2, {"$nonfinite_float": "-inf"}]}') == {
        'a': 1, 'b': [1, 2, {'$nonfinite_float': '-inf'}],
    }


@pytest.mark.filterwarnings('ignore:.*LeafSpec.*:FutureWarning')
def test_collecting_dynamic_facts_preserves_symbolic_shapes():
    import torch

    from pt2_export_core.opgraph import collect_ops

    class Model(torch.nn.Module):
        def forward(self, value):
            return value.sin() + value

    exported = torch.export.export(
        Model(), (torch.randn(2, 16),), strict=False,
        dynamic_shapes=({0: torch.export.Dim('batch', min=2, max=4),
                         1: torch.export.Dim('sequence', min=4, max=64)},),
    )
    ops, _ = collect_ops(exported)
    assert all(config['out_rank'] == 2 for _, config, _ in ops)
    functional = exported.run_decompositions(decomp_table={})
    assert len(functional.range_constraints) == 2
    value = torch.randn(3, 19)
    torch.testing.assert_close(functional.module()(value), Model()(value))


@pytest.mark.filterwarnings('ignore:.*LeafSpec.*:FutureWarning')
def test_exact_arguments_separate_constants_and_dynamic_list_elements():
    import torch
    import yaml

    from pt2_export_core import catalog
    from pt2_export_core.opgraph import collect_ops

    class Slice(torch.nn.Module):
        def __init__(self, end):
            super().__init__()
            self.end = end

        def forward(self, value):
            return value[:, :self.end].reshape(1, value.shape[0], self.end)

    def configurations(program, exact=True):
        ops, schemas = collect_ops(program, exact_symints=exact)
        return {config['end']: config for op, config, _ in ops
                if op == 'aten.slice.Tensor' and config['dim'] == 1}, ops, schemas

    sample = torch.randn(3, 16)
    first = torch.export.export(Slice(8), (sample,), strict=False)
    second = torch.export.export(Slice(12), (sample,), strict=False)
    first_configs, _, _ = configurations(first)
    second_configs, _, _ = configurations(second)
    assert first_configs[8] == {'dim': 1, 'start': 0, 'end': 8, 'step': 1,
                                'out_dtype': 'f32', 'out_rank': 2}
    assert canonical_config(first_configs[8]) != canonical_config(second_configs[12])
    assert configurations(first, exact=False)[0]['*']['step'] == '*'

    dynamic = torch.export.export(Slice(8), (sample,), strict=False,
                                  dynamic_shapes=({0: torch.export.Dim('batch', min=2, max=5)},))
    constraints = dict(dynamic.range_constraints)
    dynamic_configs, ops, schemas = configurations(dynamic)
    assert dynamic_configs == first_configs
    sizes = [config['shape'] for op, config, _ in ops if op == 'aten.reshape.default']
    assert sizes == [[1, {'$symint': True}, 8]]
    assert dynamic.range_constraints == constraints
    value = torch.randn(4, 16)
    torch.testing.assert_close(dynamic.module()(value), Slice(8)(value))
    functional = dynamic.run_decompositions(decomp_table={})
    torch.testing.assert_close(functional.module()(value), Slice(8)(value))

    matrices = {'dynamic': ops}
    rendered = catalog.render_ops_yaml(matrices, schemas, {}, catalog.ATEN,
                                       'test', '1', torch.__version__, exact_symints=True)
    assert yaml.safe_load(rendered)['ops']['reshape.default']['configs'][1]['shape'] == sizes[0]
    text = catalog.render_ops_md(matrices, schemas, {'dynamic': 'test'}, {},
                                 catalog.ATEN, 'test', '1', torch.__version__, exact_symints=True)
    assert 'shape=[1,SymInt,8]' in text
    assert 'start=0 end=8 step=1' in text
