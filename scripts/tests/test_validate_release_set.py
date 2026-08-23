import io
import json
import zipfile

import pytest
import torch

import export_pt2
from pt2_export_core.archive import ZipProfile

GENEROUS = ZipProfile(
    max_archive_bytes=50 * 2**20, max_central_directory_bytes=2 * 2**20, max_members=100,
    max_total_uncompressed=50 * 2**20, max_compression_ratio=1100, max_member_bytes=20 * 2**20,
)


def _valid_contract(images_sha256='a' * 64, pt2_member='m1.pt2'):
    tensor_map = {
        'member': 'inputs.pt', 'format': 'torch.save', 'count': 1, 'keys': ['x.jpg'],
        'dtype': 'torch.float32', 'shape': [3, 4, 4], 'device': 'cpu', 'call_signature': 'model(x)',
        'uniform': True, 'layout': 'torch.strided', 'strides': [16, 4, 1], 'storage_offset': 0,
        'is_contiguous': True,
    }
    return {
        'schema_version': 1,
        'graph': {'pt2_member': pt2_member, 'input_size': [1, 3, 4, 4]},
        'images_asset_sha256': images_sha256,
        'inputs': {**tensor_map, 'member': 'inputs.pt'},
        'outputs': {**tensor_map, 'member': 'outputs.pt', 'keys_order': 'identical to inputs',
                    'semantics': 'raw pre-softmax logits'},
        'expected': {'member': 'expected.json', 'atol': 1e-4, 'rtol': 1e-3,
                     'nonfinite_policy': 'x', 'tie_break': 'x'},
        'preprocessing': {'member': 'preprocessing.json', 'provenance': {}},
        'classes': {'count': 1000},
    }


def _make_release_zip(path, *, contract=None, inputs=None, outputs=None, pt2_member='m1.pt2'):
    contract = contract if contract is not None else _valid_contract(pt2_member=pt2_member)
    tensors = {'x.jpg': torch.zeros(3, 4, 4)}
    inputs = tensors if inputs is None else inputs
    outputs = tensors if outputs is None else outputs

    inputs_buf = io.BytesIO()
    torch.save(inputs, inputs_buf)
    outputs_buf = io.BytesIO()
    torch.save(outputs, outputs_buf)

    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(pt2_member, b'not-a-real-pt2')
        z.writestr('contract.json', json.dumps(contract))
        z.writestr('inputs.pt', inputs_buf.getvalue())
        z.writestr('outputs.pt', outputs_buf.getvalue())
        z.writestr('preprocessing.json', b'{}')
        z.writestr('expected.json', b'{}')


def test_valid_entry_passes(tmp_path):
    path = tmp_path / 'm1.zip'
    _make_release_zip(str(path))
    export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc=None)


def test_images_sha256_mismatch_against_manifest_caught(tmp_path):
    path = tmp_path / 'm1.zip'
    _make_release_zip(str(path), contract=_valid_contract(images_sha256='a' * 64))
    manifest_doc = {'images': {'sha256': 'b' * 64}}
    with pytest.raises(ValueError, match='images_asset_sha256'):
        export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc)


def test_pt2_member_not_in_archive_caught(tmp_path):
    path = tmp_path / 'm1.zip'
    contract = _valid_contract(pt2_member='does-not-exist.pt2')
    _make_release_zip(str(path), contract=contract, pt2_member='m1.pt2')
    with pytest.raises(ValueError, match='pt2_member'):
        export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc=None)


def test_non_dict_inputs_pt_rejected(tmp_path):
    path = tmp_path / 'm1.zip'
    _make_release_zip(str(path), inputs=torch.zeros(3, 4, 4))  # a bare tensor, not a dict
    with pytest.raises(ValueError, match='expected a dict of tensors'):
        export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc=None)


def test_keys_mismatch_between_contract_and_actual_tensors_caught(tmp_path):
    path = tmp_path / 'm1.zip'
    contract = _valid_contract()
    contract['inputs']['keys'] = ['different.jpg']
    _make_release_zip(str(path), contract=contract)
    with pytest.raises(ValueError, match='do not match'):
        export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc=None)


def test_contract_failing_its_own_schema_is_caught(tmp_path):
    path = tmp_path / 'm1.zip'
    contract = _valid_contract()
    del contract['classes']  # required field
    _make_release_zip(str(path), contract=contract)
    with pytest.raises(Exception):
        export_pt2._validate_release_set_entry(str(path), 'm1', GENEROUS, manifest_doc=None)
