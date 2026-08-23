import os

import pytest
from jsonschema.exceptions import ValidationError

from pt2_export_core.schema_validate import validate_document

SCHEMAS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'schemas'))

SHA = 'a' * 64


def valid_op_facts():
    return {
        'schema_version': 1,
        'torch_version': '2.9.0',
        'dialect': 'functional_aten',
        'model_json_sha256': SHA,
        'ops': [
            ['aten.convolution.default', {'stride': [1, 1]}, 20],
            ['aten.pad.default', {'value': {'$nonfinite_float': '-inf'}}, 3],
        ],
        'schemas': {'aten.convolution.default': 'convolution(Tensor input) -> Tensor'},
    }


def valid_contract():
    tensor_map = {
        'member': 'inputs.pt', 'format': 'torch.save', 'count': 5,
        'keys': ['a.jpg', 'b.jpg'], 'dtype': 'f32', 'shape': [3, 224, 224],
        'device': 'cpu', 'call_signature': 'model(x)', 'uniform': True,
        'layout': 'torch.strided', 'strides': [150528, 224, 1], 'storage_offset': 0,
        'is_contiguous': True,
    }
    return {
        'schema_version': 1,
        'graph': {'pt2_member': 'model.pt2', 'input_size': [1, 3, 224, 224]},
        'images_asset_sha256': SHA,
        'inputs': {**tensor_map, 'member': 'inputs.pt'},
        'outputs': {**tensor_map, 'member': 'outputs.pt', 'keys_order': 'identical to inputs',
                    'semantics': 'raw pre-softmax logits'},
        'expected': {'member': 'expected.json', 'atol': 1e-4, 'rtol': 1e-3,
                     'nonfinite_policy': 'any non-finite position is an automatic failure',
                     'tie_break': '(-logit_value, class_index)'},
        'preprocessing': {'member': 'preprocessing.json', 'provenance': {'timm_version': '1.0'}},
        'classes': {'count': 1000},
    }


def valid_manifest():
    return {
        'schema_version': 1,
        'scope': 'full',
        'producer': {
            'repo': 'org/repo', 'tag': 'v0.0.4', 'commit': 'b' * 40,
            'commit_timestamp': '2026-08-21T00:00:00+00:00',
            'timm_version': '1.0', 'torch_version': '2.9.0',
        },
        'selected_models': {'count': 1, 'names': ['convit_tiny']},
        'release_models': {'count': 1, 'names': ['convit_tiny']},
        'images': {'url': 'https://example.invalid/images.zip', 'sha256': SHA, 'bytes': 10},
        'models': {
            'convit_tiny': {
                'status': 'current',
                'archive': {'url': 'https://example.invalid/convit_tiny.zip', 'sha256': SHA,
                            'bytes': 100, 'members': ['convit_tiny.pt2', 'contract.json']},
            },
        },
        'retired': {
            'resnet18': {'removed_in': 'v0.0.2', 'reason': 'weight cap', 'migrate_to': 'resnet18d'},
        },
    }


def valid_catalogue():
    return {
        'schema_version': 1,
        'scope': 'full',
        'default_model': 'convit_tiny',
        'models': {
            'convit_tiny': {
                'display_name': 'ConViT Tiny', 'aliases': [], 'deprecated': False,
                'graph_json_url': 'https://raw.githubusercontent.com/org/repo/' + 'b' * 40 + '/models/convit_tiny/models/model.json',
            },
        },
    }


def valid_compat_report(scope='full'):
    doc = {
        'schema_version': 1,
        'scope': scope,
        'selected_models': {'count': 1, 'names': ['convit_tiny']},
        'release_models': {'count': 1, 'names': ['convit_tiny']},
        'backends': ['interpreter', 'aot_inductor_cpu'],
        'models': {
            'convit_tiny': {
                'classification': 'runnable',
                'graph': {'histogram': {'aten.convolution.default': 20}, 'config_variants': 1,
                          'dtypes': ['f32']},
                'backends': {
                    'interpreter': {'result': 'passed'},
                    'aot_inductor_cpu': {'result': 'passed'},
                },
            },
        },
    }
    if scope == 'smoke':
        doc['attempted_models'] = {'count': 1, 'names': ['convit_tiny']}
    return doc


@pytest.mark.parametrize('schema_name,document', [
    ('op_facts', valid_op_facts()),
    ('contract', valid_contract()),
    ('manifest', valid_manifest()),
    ('catalogue', valid_catalogue()),
    ('compat-report', valid_compat_report('full')),
    ('compat-report', valid_compat_report('smoke')),
])
def test_valid_documents_pass(schema_name, document):
    validate_document(document, schema_name, SCHEMAS_DIR)


def test_op_facts_rejects_bad_sha256():
    doc = valid_op_facts()
    doc['model_json_sha256'] = 'not-a-hash'
    with pytest.raises(ValidationError):
        validate_document(doc, 'op_facts', SCHEMAS_DIR)


def test_manifest_rejects_smoke_scope():
    doc = valid_manifest()
    doc['scope'] = 'smoke'
    with pytest.raises(ValidationError):
        validate_document(doc, 'manifest', SCHEMAS_DIR)


def test_manifest_rejects_missing_retired_migrate_to_field():
    doc = valid_manifest()
    del doc['retired']['resnet18']['migrate_to']
    with pytest.raises(ValidationError):
        validate_document(doc, 'manifest', SCHEMAS_DIR)


def test_compat_report_smoke_requires_attempted_models():
    doc = valid_compat_report('smoke')
    del doc['attempted_models']
    with pytest.raises(ValidationError):
        validate_document(doc, 'compat-report', SCHEMAS_DIR)


def test_compat_report_rejects_unknown_classification():
    doc = valid_compat_report('full')
    doc['models']['convit_tiny']['classification'] = 'maybe'
    with pytest.raises(ValidationError):
        validate_document(doc, 'compat-report', SCHEMAS_DIR)


def test_compat_report_not_attempted_requires_reason():
    doc = valid_compat_report('smoke')
    doc['models']['convit_tiny']['backends']['interpreter'] = {'result': 'not_attempted'}
    with pytest.raises(ValidationError):
        validate_document(doc, 'compat-report', SCHEMAS_DIR)


def test_contract_uniform_false_requires_per_key():
    doc = valid_contract()
    doc['inputs']['uniform'] = False
    with pytest.raises(ValidationError):
        validate_document(doc, 'contract', SCHEMAS_DIR)
