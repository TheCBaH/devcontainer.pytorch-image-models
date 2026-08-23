import hashlib
import json
import os

import pytest

from pt2_export_core import compat
from pt2_export_core.schema_validate import validate_document

SCHEMAS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'schemas'))


def _write_model(models_dir, name, ops):
    model_dir = os.path.join(models_dir, name, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_json_bytes = json.dumps({'graph_module': {'graph': {'nodes': []}}}).encode()
    with open(os.path.join(model_dir, 'model.json'), 'wb') as f:
        f.write(model_json_bytes)
    op_facts = {
        'schema_version': 1, 'torch_version': '2.9.0', 'dialect': 'functional_aten',
        'model_json_sha256': hashlib.sha256(model_json_bytes).hexdigest(),
        'ops': ops, 'schemas': {},
    }
    with open(os.path.join(model_dir, 'op_facts.json'), 'w') as f:
        json.dump(op_facts, f)
    return model_json_bytes


# ---------------------------------------------------------------------------- static_graph_facts


def test_static_graph_facts_histogram_and_dtypes(tmp_path):
    _write_model(str(tmp_path), 'm1', [
        ['aten.convolution.default', {'out_dtype': 'f32'}, 20],
        ['aten.relu.default', {'out_dtype': 'f32'}, 5],
        ['aten.linear.default', {'out_dtype': 'i64'}, 1],
    ])
    facts = compat.static_graph_facts(str(tmp_path), 'm1')
    assert facts['histogram'] == {'aten.convolution.default': 20, 'aten.relu.default': 5,
                                 'aten.linear.default': 1}
    assert facts['config_variants'] == 3
    assert facts['dtypes'] == ['f32', 'i64']


def test_static_graph_facts_missing_sidecar_raises(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'm1', 'models'))
    with open(os.path.join(tmp_path, 'm1', 'models', 'model.json'), 'wb') as f:
        f.write(b'{}')
    with pytest.raises(FileNotFoundError):
        compat.static_graph_facts(str(tmp_path), 'm1')


def test_static_graph_facts_rejects_stale_sha256(tmp_path):
    _write_model(str(tmp_path), 'm1', [['aten.relu.default', {}, 1]])
    # Mutate model.json without touching op_facts.json -- the digest binding must catch this.
    model_json_path = os.path.join(tmp_path, 'm1', 'models', 'model.json')
    with open(model_json_path, 'wb') as f:
        f.write(b'{"graph_module": {"graph": {"nodes": [], "extra": true}}}')
    with pytest.raises(ValueError, match='model_json_sha256'):
        compat.static_graph_facts(str(tmp_path), 'm1')


# ---------------------------------------------------------------------------- classify


def test_classify_graph_only():
    classification, backends = compat.classify(is_release_tier=False, attempted=False)
    assert classification == 'graph_only'
    assert backends['interpreter'] == {'result': 'not_attempted', 'reason': 'not_release_tier'}
    assert backends['aot_inductor_cpu'] == {'result': 'not_attempted', 'reason': 'not_release_tier'}


def test_classify_unverified_outside_smoke_scope():
    classification, backends = compat.classify(is_release_tier=True, attempted=False)
    assert classification == 'unverified'
    assert backends['interpreter']['reason'] == 'outside_smoke_scope'
    assert backends['aot_inductor_cpu']['reason'] == 'outside_smoke_scope'


def test_classify_unsupported_when_interpreter_fails():
    classification, backends = compat.classify(
        is_release_tier=True, attempted=True,
        pack_result={'status': 'failed', 'error': "image 'a.jpg' (0/2 run): boom"})
    assert classification == 'unsupported'
    assert backends['interpreter']['result'] == 'failed'
    assert 'boom' in backends['interpreter']['error']
    assert backends['aot_inductor_cpu'] == {'result': 'not_attempted', 'reason': 'prior_backend_failed'}


def test_classify_runnable_with_no_aoti_attempt_recorded():
    classification, backends = compat.classify(
        is_release_tier=True, attempted=True, pack_result={'status': 'ok'})
    assert classification == 'runnable'
    assert backends['interpreter'] == {'result': 'passed'}
    assert backends['aot_inductor_cpu'] == {'result': 'not_attempted', 'reason': 'prior_backend_failed'}


def test_classify_runnable_with_aoti_pass_and_fail():
    _, backends_pass = compat.classify(
        is_release_tier=True, attempted=True,
        pack_result={'status': 'ok', 'aoti': {'status': 'ok', 'environment': {'torch': '2.9.0'}}})
    assert backends_pass['aot_inductor_cpu']['result'] == 'passed'

    _, backends_fail = compat.classify(
        is_release_tier=True, attempted=True,
        pack_result={'status': 'ok', 'aoti': {'status': 'failed', 'error': 'unsupported op X',
                                             'hint': 'aten.foo.default'}})
    assert backends_fail['aot_inductor_cpu']['result'] == 'failed'
    assert backends_fail['aot_inductor_cpu']['hint'] == 'aten.foo.default'
    # aoti failure never changes the pack's own (interpreter) success.
    assert backends_fail['interpreter'] == {'result': 'passed'}


# ---------------------------------------------------------------------------- render_compat_report


def test_render_compat_report_full_scope_validates(tmp_path):
    _write_model(str(tmp_path), 'released', [['aten.relu.default', {'out_dtype': 'f32'}, 1]])
    _write_model(str(tmp_path), 'graph_only', [['aten.relu.default', {'out_dtype': 'f32'}, 1]])
    doc = compat.render_compat_report(
        scope='full', selected_names={'released', 'graph_only'}, release_names={'released'},
        attempted_names={'released'}, models_dir=str(tmp_path),
        pack_results={'released': {'status': 'ok'}},
    )
    assert doc['models']['released']['classification'] == 'runnable'
    assert doc['models']['graph_only']['classification'] == 'graph_only'
    assert 'attempted_models' not in doc
    validate_document(doc, 'compat-report', SCHEMAS_DIR)


def test_render_compat_report_smoke_scope_marks_unattempted_release_models_unverified(tmp_path):
    _write_model(str(tmp_path), 'attempted', [['aten.relu.default', {'out_dtype': 'f32'}, 1]])
    _write_model(str(tmp_path), 'not_attempted', [['aten.relu.default', {'out_dtype': 'f32'}, 1]])
    doc = compat.render_compat_report(
        scope='smoke', selected_names={'attempted', 'not_attempted'},
        release_names={'attempted', 'not_attempted'}, attempted_names={'attempted'},
        models_dir=str(tmp_path), pack_results={'attempted': {'status': 'ok'}},
    )
    assert doc['models']['attempted']['classification'] == 'runnable'
    assert doc['models']['not_attempted']['classification'] == 'unverified'
    assert doc['attempted_models'] == {'count': 1, 'names': ['attempted']}
    validate_document(doc, 'compat-report', SCHEMAS_DIR)


def test_render_compat_report_smoke_scope_can_contain_unsupported(tmp_path):
    _write_model(str(tmp_path), 'flaky', [['aten.relu.default', {'out_dtype': 'f32'}, 1]])
    doc = compat.render_compat_report(
        scope='smoke', selected_names={'flaky'}, release_names={'flaky'},
        attempted_names={'flaky'}, models_dir=str(tmp_path),
        pack_results={'flaky': {'status': 'failed', 'error': 'boom'}},
    )
    assert doc['models']['flaky']['classification'] == 'unsupported'
    validate_document(doc, 'compat-report', SCHEMAS_DIR)
