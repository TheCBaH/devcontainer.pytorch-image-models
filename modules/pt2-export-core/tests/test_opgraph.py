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
