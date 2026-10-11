import copy
import hashlib
import io
import json
import shutil
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from timm.data import create_transform

import evaluation_contract as contract
import input_assets as assets


def test_canonical_hash_ignores_storage_and_preserves_bits():
    backing = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    view = backing[1:4, ::2]
    assert assets.tensor_descriptor(view) == assets.tensor_descriptor(view.clone().contiguous())
    raw = np.array([1.0, -0.0, 3.5], dtype='>f4')
    assert assets.canonical_array_bytes(raw) == assets.canonical_array_bytes(raw.astype('<f4'))
    assert assets.tensor_descriptor(torch.tensor([-0.0])) != assets.tensor_descriptor(torch.tensor([0.0]))


@pytest.mark.parametrize('value', [torch.tensor([float('nan')]), torch.tensor([float('inf')]),
                                 torch.zeros(4, dtype=torch.float64)])
def test_unsupported_or_nonfinite_tensor_rejected(value):
    with pytest.raises(ValueError):
        assets.tensor_descriptor(value)


def graph_bytes():
    tensors = {k: {'dtype': 7, 'sizes': [{'as_int': x} for x in dims]}
               for k, dims in [('first', [1, 3]), ('second', [1, 2]), ('logits', [1, 4])]}
    return assets.json_bytes({'graph_module': {'signature': {
        'input_specs': [{'parameter': {'arg': {'name': 'weight'}}},
                        *[{'user_input': {'arg': {'as_tensor': {'name': n}}}}
                          for n in ['first', 'second']]],
        'output_specs': [{'user_output': {'arg': {'as_tensor': {'name': 'logits'}}}}]},
        'graph': {'tensor_values': tensors}}})


def fixture_contract():
    inputs = {'z-key': torch.arange(3, dtype=torch.float32),
              'a-key': torch.arange(2, dtype=torch.float32)}
    outputs = {'ref-key': torch.arange(4, dtype=torch.float32)}
    c, cases = contract.build_contract('two', graph_bytes(), inputs, outputs,
        decoded_sha256='a' * 64, geometry={'photo': []},
        input_specs={'second': ('aux', {'op': 'unsqueeze', 'axis': 0}),
                     'first': ('image', {'op': 'unsqueeze', 'axis': 0})},
        output_specs={'logits': ('scores', [{'op': 'select', 'axis': 0, 'index': 0},
                                          {'op': 'cast', 'dtype': 'float32'}])},
        case_mappings=[{'id': 'case', 'decoded_image_key': 'photo',
                        'inputs': {'aux': 'a-key', 'image': 'z-key'},
                        'outputs': {'scores': 'ref-key'}}])
    return c, cases, inputs, outputs


def test_two_inputs_follow_signature_not_tensor_or_spec_order():
    c, cases, inputs, outputs = fixture_contract()
    assert [b['graph_input'] for b in c['inputs']['bindings']] == ['first', 'second']
    assert cases['cases'][0]['inputs']['image']['tensor_key'] == 'z-key'
    contract.validate_contract(c, cases, graph_bytes(), inputs, outputs, decoded_keys={'photo'})
    assert torch.equal(contract.output_value(torch.arange(4).reshape(1, 4),
                        c['outputs']['bindings'][0]['transform']), outputs['ref-key'])


@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'axis', 'dtype', 'shape', 'key',
                                     'operation', 'case', 'descriptor', 'graph', 'image', 'byte_order',
                                     'output_axis', 'output_dtype', 'output_shape'])
def test_invalid_bindings_and_cases_rejected(mutation):
    c, cases, inputs, outputs = fixture_contract()
    binding = c['inputs']['bindings'][0]
    if mutation == 'omit': c['inputs']['bindings'].pop()
    if mutation == 'duplicate': c['inputs']['bindings'].append(copy.deepcopy(binding))
    if mutation == 'axis': binding['transform']['axis'] = 1
    if mutation == 'dtype': binding['sample']['dtype'] = 'uint8'
    if mutation == 'shape': binding['graph_shape'] = [1, 2]
    if mutation == 'key': cases['cases'][0]['inputs']['image']['tensor_key'] = 'absent'
    if mutation == 'operation': binding['transform'] = {'op': 'guess'}
    if mutation == 'case': cases['cases'].append(copy.deepcopy(cases['cases'][0]))
    if mutation == 'descriptor': cases['cases'][0]['inputs']['image']['tensor']['sha256'] = 'b' * 64
    if mutation == 'graph': c['graph']['sha256'] = 'b' * 64
    if mutation == 'image': cases['cases'][0]['decoded_image_key'] = 'absent'
    if mutation == 'byte_order': cases['cases'][0]['inputs']['image']['tensor']['byte_order'] = 'big'
    if mutation == 'output_axis': c['outputs']['bindings'][0]['transform'][0]['axis'] = 1
    if mutation == 'output_dtype': c['outputs']['bindings'][0]['sample']['dtype'] = 'uint8'
    if mutation == 'output_shape': c['outputs']['bindings'][0]['sample']['shape'] = [5]
    with pytest.raises(Exception):
        contract.validate_contract(c, cases, graph_bytes(), inputs, outputs, decoded_keys={'photo'})


def patterned(width, height):
    y, x = np.mgrid[:height, :width]
    return Image.fromarray(np.stack([(x * 17 + y * 5) % 256, (x * 3 + y * 31) % 256,
                                     (x * 41 + y * 7) % 256], axis=2).astype(np.uint8))


def decoded_fixture(tmp_path):
    path = tmp_path / 'images.zip'
    originals = {}
    with zipfile.ZipFile(path, 'w') as z:
        for name, dims in [('portrait.jpg', (39, 53)), ('landscape.jpg', (61, 37))]:
            buffer = io.BytesIO()
            exif = Image.Exif()
            exif[274] = 6
            patterned(*dims).save(buffer, format='JPEG', exif=exif)
            data = buffer.getvalue()
            z.writestr('images/' + name, data)
            originals[name] = Image.open(io.BytesIO(data)).convert('RGB')
        buffer = io.BytesIO()
        patterned(35, 43).convert('L').save(buffer, format='PNG')
        z.writestr('images/gray.png', buffer.getvalue())
        originals['gray.png'] = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
        z.writestr('SOURCES.md', 'licensed patterned fixtures')
        z.writestr('labels/classes.txt', 'class 0\n')
    images_path, metadata_path = tmp_path / 'images.pt', tmp_path / 'images-metadata.json'
    images = assets.decode_images(path, images_path, metadata_path, assets.sha256_file(path))
    return images, originals, images_path, metadata_path


@pytest.mark.parametrize('mode', ['center', 'squash', 'border'])
@pytest.mark.parametrize('interpolation', ['bilinear', 'bicubic'])
def test_decoded_roundtrip_pipeline_and_observed_geometry(tmp_path, mode, interpolation):
    images, originals, _, metadata_path = decoded_fixture(tmp_path)
    metadata = json.loads(metadata_path.read_bytes())
    assert metadata['decoding']['exif_orientation'] == 'not_applied'
    assert metadata['samples']['portrait.jpg']['tensor']['shape'] == [3, 53, 39]
    assert metadata['attribution']['SOURCES.md'] == 'licensed patterned fixtures'
    config = {'input_size': [3, 17, 19], 'crop_pct': 0.89, 'crop_mode': mode,
              'interpolation': interpolation, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = create_transform(**config, is_training=False)
    preprocessing = {'recipe': assets.recipe(config, transform)}
    reproduced, geometry = assets.reproduce(images, preprocessing)
    for key, image in images.items():
        assert np.array_equal(np.asarray(assets.rgb_image(image)), np.asarray(originals[key]))
        expected = transform(originals[key])
        assert assets.tensor_descriptor(reproduced[key]) == assets.tensor_descriptor(expected)
        assert geometry[key][0]['op'] == 'resize'
        assert geometry[key][-1]['after'] == [17, 19]
    if mode == 'border':
        assert any(event['op'] == 'pad' for events in geometry.values() for event in events)


def test_decoded_image_file_and_metadata_tamper_rejected(tmp_path):
    images, _, path, meta = decoded_fixture(tmp_path)
    digest = assets.sha256_file(path)
    with pytest.raises(ValueError, match='sha256'):
        assets.load_images(path, meta, expected_sha256='b' * 64)
    doc = json.loads(meta.read_bytes())
    doc['samples']['portrait.jpg']['tensor']['sha256'] = 'b' * 64
    meta.write_bytes(assets.json_bytes(doc))
    with pytest.raises(ValueError, match='descriptor'):
        assets.load_images(path, meta, expected_sha256=digest)


def test_source_digest_and_missing_sample_rejected(tmp_path):
    _, _, path, meta = decoded_fixture(tmp_path)
    with pytest.raises(ValueError, match='source'):
        assets.decode_images(tmp_path / 'images.zip', path, meta, 'b' * 64)
    images = assets.load_tensors(path)
    images.pop('portrait.jpg')
    path.write_bytes(assets.tensor_bytes(images))
    doc = json.loads(meta.read_bytes())
    doc['images_sha256'] = assets.sha256_file(path)
    meta.write_bytes(assets.json_bytes(doc))
    with pytest.raises(ValueError, match='keys'):
        assets.load_images(path, meta, expected_sha256=doc['images_sha256'])
