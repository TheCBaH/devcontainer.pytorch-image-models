import argparse
import json

import pytest
import yaml

import export_pt2
import weights_map

REVISION = 'b' * 40


def _config(tensors):
    return {'config': {name: {'path_name': f'tensor_{i}', 'is_param': True, 'use_pickle': False,
                              'tensor_meta': {'dtype': dtype,
                                              'sizes': [{'as_int': s} for s in shape]}}
                       for i, (name, (shape, dtype)) in enumerate(tensors.items())}}


def _model_dir(root, name, weights, constants=None):
    model_dir = root / name
    (model_dir / 'data' / 'weights').mkdir(parents=True)
    (model_dir / 'models').mkdir()
    (model_dir / 'data' / 'weights' / 'model_weights_config.json').write_text(json.dumps(_config(weights)))
    if constants:
        (model_dir / 'data' / 'constants').mkdir()
        (model_dir / 'data' / 'constants' / 'model_constants_config.json').write_text(
            json.dumps(_config(constants)))
    return model_dir


def _document(repo_id, tensors, unmapped=()):
    return {
        'schema_version': 1,
        'source': {'repo_id': repo_id, 'revision': REVISION, 'filename': weights_map.FILENAME,
                   'url': f'https://huggingface.co/{repo_id}/resolve/{REVISION}/model.safetensors',
                   'sha256': 'c' * 64, 'size': 1234},
        'tensors': {n: {'key': n, 'dtype': d, 'shape': s} for n, (s, d) in tensors.items()},
        'unmapped': list(unmapped),
    }


WEIGHTS = {'conv.weight': ([8, 3, 3, 3], 7), 'bn.num_batches_tracked': ([], 5)}
HEADER = {'conv.weight': ([8, 3, 3, 3], 'F32'), 'bn.num_batches_tracked': ([], 'I64')}


def test_every_weight_maps_by_name():
    graph = {n: (s, weights_map.SCALAR_TYPES[d]) for n, (s, d) in WEIGHTS.items()}
    tensors, unmapped = weights_map.map_tensors(graph, {}, HEADER)
    assert tensors == {'bn.num_batches_tracked': {'key': 'bn.num_batches_tracked', 'dtype': 'I64', 'shape': []},
                       'conv.weight': {'key': 'conv.weight', 'dtype': 'F32', 'shape': [8, 3, 3, 3]}}
    assert unmapped == []


def test_cast_graph_takes_fp32_checkpoint_but_not_int_to_float():
    tensors, _ = weights_map.map_tensors({'conv.weight': ([8, 3, 3, 3], 'F16')}, {}, HEADER)
    assert tensors['conv.weight']['dtype'] == 'F32'
    with pytest.raises(ValueError, match='dtype I64 != F32'):
        weights_map.map_tensors({'bn.num_batches_tracked': ([], 'F32')}, {}, HEADER)


@pytest.mark.parametrize('graph,match', [
    ({'head.weight': ([10, 8], 'F32')}, 'head.weight: not in checkpoint'),
    ({'conv.weight': ([16, 3, 3, 3], 'F32')}, r'conv.weight: shape \[8, 3, 3, 3\] != \[16, 3, 3, 3\]'),
])
def test_unmappable_weight_rejects_the_model(graph, match):
    with pytest.raises(ValueError, match=match):
        weights_map.map_tensors(graph, {}, HEADER)


def test_autocast_graph_names_map_to_keys_less_the_wrapper_prefix():
    graph = {'model.conv.weight': ([8, 3, 3, 3], 'F32'), 'conv.weight': ([8, 3, 3, 3], 'F32')}
    with pytest.raises(ValueError, match='^conv.weight: not in checkpoint'):
        weights_map.map_tensors(graph, {}, HEADER, 'model.')
    tensors, _ = weights_map.map_tensors({'model.conv.weight': ([8, 3, 3, 3], 'F32')}, {}, HEADER, 'model.')
    assert tensors == {'model.conv.weight': {'key': 'conv.weight', 'dtype': 'F32', 'shape': [8, 3, 3, 3]}}


def test_constants_map_only_where_the_checkpoint_has_them():
    header = {**HEADER, 'attn.relative_position_index': ([49, 49], 'I64')}
    constants = {'attn.relative_position_index': ([49, 49], 'I64'), 'attn_mask': ([4, 49, 49], 'F32')}
    tensors, unmapped = weights_map.map_tensors({}, constants, header)
    assert list(tensors) == ['attn.relative_position_index']
    assert unmapped == ['attn_mask']


def _tree(tmp_path, manifest_models):
    manifest = tmp_path / 'models-selected.yaml'
    manifest.write_text(yaml.safe_dump({'models': manifest_models}))
    return str(manifest)


def _verify(tmp_path, manifest, unmapped=None, prefix=''):
    unmapped_path = tmp_path / 'safetensors-unmapped.yaml'
    unmapped_path.write_text(yaml.safe_dump({'models': unmapped or {}}))
    args = argparse.Namespace(tree=[(manifest, str(tmp_path / 'models'), prefix)],
                              unmapped=str(unmapped_path))
    return weights_map.cmd_verify(args)


def test_verify_accepts_a_consistent_map(tmp_path):
    model_dir = _model_dir(tmp_path / 'models', 'm', WEIGHTS, {'attn_mask': ([4], 7)})
    (model_dir / weights_map.WEIGHTS_MAP).write_text(
        weights_map.render(_document('timm/m.tag', HEADER, ['attn_mask'])))
    manifest = _tree(tmp_path, {'m': {'hf_hub_id': 'timm/m.tag'}, 'no_weights': {}})
    assert _verify(tmp_path, manifest) == 0


def test_verify_accepts_a_prefixed_autocast_map(tmp_path):
    model_dir = _model_dir(tmp_path / 'models', 'm', {f'model.{n}': v for n, v in WEIGHTS.items()})
    document = _document('timm/m.tag', HEADER)
    document['tensors'] = {f'model.{n}': t for n, t in document['tensors'].items()}
    (model_dir / weights_map.WEIGHTS_MAP).write_text(weights_map.render(document))
    manifest = _tree(tmp_path, {'m': {'hf_hub_id': 'timm/m.tag'}})
    assert _verify(tmp_path, manifest, prefix='model.') == 0
    assert _verify(tmp_path, manifest) == 1


def test_verify_catches_a_graph_that_moved_under_the_map(tmp_path, capsys):
    model_dir = _model_dir(tmp_path / 'models', 'm', {**WEIGHTS, 'fc.weight': ([10, 8], 7)})
    (model_dir / weights_map.WEIGHTS_MAP).write_text(weights_map.render(_document('timm/m.tag', HEADER)))
    assert _verify(tmp_path, _tree(tmp_path, {'m': {'hf_hub_id': 'timm/m.tag'}})) == 1
    assert 'fc.weight: not in checkpoint' in capsys.readouterr().err


def test_verify_catches_a_wrong_repo_and_a_url_off_its_revision(tmp_path, capsys):
    model_dir = _model_dir(tmp_path / 'models', 'm', WEIGHTS)
    document = _document('timm/m.old', HEADER)
    document['source']['url'] = document['source']['url'].replace(REVISION, 'main')
    (model_dir / weights_map.WEIGHTS_MAP).write_text(weights_map.render(document))
    assert _verify(tmp_path, _tree(tmp_path, {'m': {'hf_hub_id': 'timm/m.tag'}})) == 1
    err = capsys.readouterr().err
    assert 'repo_id timm/m.old != manifest hf_hub_id timm/m.tag' in err
    assert 'does not match repo_id/filename/revision' in err


def test_verify_requires_a_map_or_an_unmapped_entry_but_not_both(tmp_path, capsys):
    _model_dir(tmp_path / 'models', 'm', WEIGHTS)
    manifest = _tree(tmp_path, {'m': {'hf_hub_id': 'timm/m.tag'}})
    assert _verify(tmp_path, manifest) == 1
    assert 'no models/safetensors.json and not listed as unmapped' in capsys.readouterr().err
    assert _verify(tmp_path, manifest, {'m': 'no model.safetensors in timm/m.tag'}) == 0

    (tmp_path / 'models' / 'm' / weights_map.WEIGHTS_MAP).write_text(
        weights_map.render(_document('timm/m.tag', HEADER)))
    assert _verify(tmp_path, manifest, {'m': 'no model.safetensors in timm/m.tag'}) == 1
    assert 'but is listed as unmapped' in capsys.readouterr().err


def test_rebuild_keeps_the_weights_map(tmp_path):
    models_dir = tmp_path / 'models'
    _model_dir(models_dir, 'm', WEIGHTS)
    (models_dir / 'm' / weights_map.WEIGHTS_MAP).write_text('{"kept": true}\n')
    staged = _model_dir(models_dir, 'm.building-1', WEIGHTS)
    export_pt2._carry_over_weights_map(str(models_dir), 'm', str(staged))
    export_pt2._swap_into_place(str(models_dir), 'm', str(staged))
    assert (models_dir / 'm' / weights_map.WEIGHTS_MAP).read_text() == '{"kept": true}\n'
