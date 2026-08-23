import io
import os
import zipfile

import pytest

from pt2_export_core.archive import ZipProfile, compare_embedded_graph

GENEROUS = ZipProfile(
    max_archive_bytes=50 * 2**20,
    max_central_directory_bytes=2 * 2**20,
    max_members=100,
    max_total_uncompressed=50 * 2**20,
    max_compression_ratio=200,
    max_member_bytes=20 * 2**20,
)


def _make_inner_pt2_bytes(model_json_bytes, root='m1'):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr(f'{root}/models/model.json', model_json_bytes)
        z.writestr(f'{root}/data/weights/model_weights_config.json', b'{}')
    return buf.getvalue()


def _make_outer_release_zip(path, name, inner_pt2_bytes):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{name}.pt2', inner_pt2_bytes)
        z.writestr('preprocessing.json', b'{}')
        z.writestr('expected.json', b'{}')
        z.writestr('inputs.pt', b'')
        z.writestr('outputs.pt', b'')


def _write_committed(models_dir, name, model_json_bytes):
    committed_dir = os.path.join(models_dir, name, 'models')
    os.makedirs(committed_dir, exist_ok=True)
    with open(os.path.join(committed_dir, 'model.json'), 'wb') as f:
        f.write(model_json_bytes)


def test_matching_graph_passes(tmp_path):
    models_dir = tmp_path / 'models'
    model_json = b'{"graph_module": {"nodes": []}}'
    _write_committed(str(models_dir), 'm1', model_json)
    release_zip = tmp_path / 'm1.zip'
    _make_outer_release_zip(str(release_zip), 'm1', _make_inner_pt2_bytes(model_json))

    compare_embedded_graph(str(release_zip), 'm1', str(models_dir), GENEROUS)  # no raise


def test_mismatched_graph_raises_naming_both_paths(tmp_path):
    models_dir = tmp_path / 'models'
    _write_committed(str(models_dir), 'm1', b'{"committed": true}')
    release_zip = tmp_path / 'm1.zip'
    _make_outer_release_zip(str(release_zip), 'm1', _make_inner_pt2_bytes(b'{"embedded": true}'))

    with pytest.raises(ValueError, match=r'm1\.zip!m1\.pt2'):
        compare_embedded_graph(str(release_zip), 'm1', str(models_dir), GENEROUS)


def test_missing_committed_graph_raises(tmp_path):
    models_dir = tmp_path / 'models'
    release_zip = tmp_path / 'm1.zip'
    _make_outer_release_zip(str(release_zip), 'm1', _make_inner_pt2_bytes(b'{}'))
    with pytest.raises(ValueError, match='no committed graph'):
        compare_embedded_graph(str(release_zip), 'm1', str(models_dir), GENEROUS)


def test_multi_root_inner_pt2_rejected(tmp_path):
    models_dir = tmp_path / 'models'
    model_json = b'{}'
    _write_committed(str(models_dir), 'm1', model_json)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('rootA/models/model.json', model_json)
        z.writestr('rootB/models/model.json', model_json)
    release_zip = tmp_path / 'm1.zip'
    _make_outer_release_zip(str(release_zip), 'm1', buf.getvalue())
    with pytest.raises(ValueError, match='single top-level directory'):
        compare_embedded_graph(str(release_zip), 'm1', str(models_dir), GENEROUS)


def test_missing_pt2_member_in_outer_zip_raises(tmp_path):
    models_dir = tmp_path / 'models'
    _write_committed(str(models_dir), 'm1', b'{}')
    release_zip = tmp_path / 'm1.zip'
    with zipfile.ZipFile(str(release_zip), 'w') as z:
        z.writestr('preprocessing.json', b'{}')
    with pytest.raises(ValueError, match='no .m1\\.pt2. member'):
        compare_embedded_graph(str(release_zip), 'm1', str(models_dir), GENEROUS)
