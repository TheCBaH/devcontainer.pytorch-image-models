import hashlib
import json
import os
import warnings
import zipfile

import pytest

import export_pt2
from pt2_export_core.archive import SafeZipError, release_pt2_profile

REPO_ROOT = export_pt2.REPO_ROOT
MODELS_SELECTED = os.path.join(REPO_ROOT, 'models-selected.yaml')


# ---------------------------------------------------------------------------- staging helpers


def test_clean_stale_build_artifacts_removes_only_matching_name(tmp_path):
    (tmp_path / 'foo.building-111').mkdir()
    (tmp_path / 'foo.stale-222').mkdir()
    (tmp_path / 'bar.building-333').mkdir()
    (tmp_path / 'foo').mkdir()  # the live directory itself must never be touched by cleanup

    export_pt2._clean_stale_build_artifacts(str(tmp_path), 'foo')

    remaining = set(os.listdir(tmp_path))
    assert remaining == {'bar.building-333', 'foo'}


def test_clean_stale_build_artifacts_on_missing_models_dir_is_a_noop(tmp_path):
    export_pt2._clean_stale_build_artifacts(str(tmp_path / 'does-not-exist'), 'foo')


def test_swap_into_place_first_build(tmp_path):
    staged = tmp_path / 'foo.building-1'
    staged.mkdir()
    (staged / 'marker.txt').write_text('new')

    export_pt2._swap_into_place(str(tmp_path), 'foo', str(staged))

    assert (tmp_path / 'foo' / 'marker.txt').read_text() == 'new'
    assert not staged.exists()
    assert not any(n.startswith('foo.stale-') for n in os.listdir(tmp_path))


def test_swap_into_place_replaces_existing_live_directory(tmp_path):
    live = tmp_path / 'foo'
    live.mkdir()
    (live / 'marker.txt').write_text('old')
    staged = tmp_path / 'foo.building-1'
    staged.mkdir()
    (staged / 'marker.txt').write_text('new')

    export_pt2._swap_into_place(str(tmp_path), 'foo', str(staged))

    assert (tmp_path / 'foo' / 'marker.txt').read_text() == 'new'
    assert not staged.exists()
    assert not any(n.startswith('foo.stale-') for n in os.listdir(tmp_path))


def test_stage_op_facts_writes_valid_sidecar(tmp_path):
    building = tmp_path / 'foo.building-1'
    (building / 'models').mkdir(parents=True)
    model_json_bytes = b'{"graph_module": {}}'
    (building / 'models' / 'model.json').write_bytes(model_json_bytes)

    convert_result = {
        'torch_version': '2.9.0',
        'ops': [['aten.convolution.default', {'stride': [1, 1]}, 3]],
        'schemas': {'aten.convolution.default': 'convolution(Tensor input) -> Tensor'},
    }
    export_pt2._stage_op_facts(str(building), convert_result)

    op_facts_path = building / 'models' / 'op_facts.json'
    assert op_facts_path.exists()
    doc = json.loads(op_facts_path.read_bytes())
    assert doc['model_json_sha256'] == hashlib.sha256(model_json_bytes).hexdigest()
    assert doc['dialect'] == 'functional_aten'
    assert doc['ops'] == convert_result['ops']


def test_stage_op_facts_rejects_empty_ops(tmp_path):
    building = tmp_path / 'foo.building-1'
    (building / 'models').mkdir(parents=True)
    (building / 'models' / 'model.json').write_bytes(b'{}')
    convert_result = {'torch_version': '2.9.0', 'ops': [], 'schemas': {}}
    with pytest.raises(ValueError, match='zero operations'):
        export_pt2._stage_op_facts(str(building), convert_result)
    assert not (building / 'models' / 'op_facts.json').exists()


# ---------------------------------------------------------------------------- images.zip validation


def _make_images_zip(path, samples, extra=()):
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in samples.items():
            z.writestr(f'images/{name}', data)
        for name, data in extra:
            z.writestr(name, data)


def test_read_images_archive_normalizes_and_hashes(tmp_path):
    path = tmp_path / 'images.zip'
    _make_images_zip(path, {'cat.jpg': b'c', 'dog.png': b'd'},
                     extra=[('labels/cat.txt', b'0'), ('SOURCES.md', b'# sources')])
    sha256, samples = export_pt2._read_images_archive(str(path))
    assert sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert samples == {'cat.jpg': 'images/cat.jpg', 'dog.png': 'images/dog.png'}


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_read_images_archive_rejects_duplicate_member(tmp_path):
    # zipfile.writestr itself warns on the second write of a name this test writes on
    # purpose, to build the fixture the assertion below actually exercises.
    path = tmp_path / 'images.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('images/cat.jpg', b'one')
        z.writestr('images/cat.jpg', b'two')
    with pytest.raises(SafeZipError, match='duplicate member'):
        export_pt2._read_images_archive(str(path))


def test_local_image_samples_filters_and_sorts(tmp_path):
    for name in ['b.jpg', 'a.png', 'notes.txt', 'c.JPEG']:
        (tmp_path / name).write_bytes(b'x')
    assert export_pt2._local_image_samples(str(tmp_path)) == ['a.png', 'b.jpg', 'c.JPEG']


# ---------------------------------------------------------------------------- cmd_pack gating


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_cmd_pack_rejects_graph_only_model_before_worker_pack_runs(tmp_path, monkeypatch):
    manifest = tmp_path / 'models-selected.yaml'
    manifest.write_text(
        'models:\n'
        '  release_model:\n'
        '    release: true\n'
        '  graph_only_model:\n'
        '    release: false\n'
    )
    called = []
    monkeypatch.setattr(export_pt2, 'worker_pack', lambda *a, **kw: called.append(1))

    args = _Args(name='graph_only_model', manifest=str(manifest), model='/nonexistent.pt2',
                images='/nonexistent', images_archive='/nonexistent.zip',
                images_sha256='0' * 64, output='/nonexistent-out.zip', max_res=224)
    with pytest.raises(SystemExit):
        export_pt2.cmd_pack(args)
    assert not called, 'worker_pack must never run for a non-release-tier name'


# ---------------------------------------------------------------------------- cmd_build / PRETRAINED_SENSITIVE_MODELS
#
# Postmortem for the fbnetc_100 release-verify failure: cmd_build always traced every model
# with pretrained=False (cheap, no download) to produce the committed models/<name>/model.json,
# while cmd_release's pack step traces release-tier models with pretrained=True. For almost
# every model that only changes weight values, not the exported graph -- but timm's
# fbnetc_100 factory does `if pretrained: kwargs.setdefault('bn_eps', ...)`, which bakes a
# different literal BatchNorm eps into the graph depending on the flag. verify-release's
# byte-comparison caught the resulting divergence at release-cut time.
#
# PRETRAINED_SENSITIVE_MODELS is the fix: names in it get pretrained=True in cmd_build too, so
# the committed graph matches what actually ships. This test pins that cmd_build's worker
# invocation actually threads --pretrained through for such a name, and leaves an ordinary
# model alone.


def test_cmd_build_forces_pretrained_for_sensitive_models(tmp_path, monkeypatch):
    manifest = tmp_path / 'models-selected.yaml'
    manifest.write_text(
        'models:\n'
        '  fbnetc_100:\n'
        '    release: true\n'
        '  ordinary_model:\n'
        '    release: true\n'
    )
    models_dir = tmp_path / 'models'

    calls = {}

    def fake_run_worker(script, argv, name, timeout, hf_home=None):
        calls[name] = argv
        return {'name': name, 'status': 'failed', 'error': 'stubbed'}

    monkeypatch.setattr(export_pt2, 'run_worker', fake_run_worker)

    args = _Args(manifest=str(manifest), models_dir=str(models_dir), build_dir=str(tmp_path / 'build'),
                keep_pt2=False, workers=1, timeout=60, hf_home=None, max_res=224, limit=None)
    export_pt2.cmd_build(args)

    assert '--pretrained' in calls['fbnetc_100']
    assert '--pretrained' not in calls['ordinary_model']


def test_pretrained_sensitive_models_are_in_the_current_manifest():
    """Catches the manifest typo/rename case where a PRETRAINED_SENSITIVE_MODELS entry
    silently stops matching anything -- cmd_build would then trace it with pretrained=False
    and nothing would notice until the next release-verify run."""
    models = export_pt2.load_manifest(MODELS_SELECTED)
    for name in export_pt2.PRETRAINED_SENSITIVE_MODELS:
        assert name in models, f'{name}: PRETRAINED_SENSITIVE_MODELS entry not in the manifest'


# ---------------------------------------------------------------------------- worker_convert / worker_pack round trip


@pytest.fixture(scope='module')
def tiny_pt2(tmp_path_factory):
    """A real, fast .pt2 for a tiny timm test architecture, pretrained=False (no network)."""
    out_dir = tmp_path_factory.mktemp('tiny_pt2')
    pt2_path = str(out_dir / 'test_vit4.pt2')
    with warnings.catch_warnings():
        # torch.export.save's pytree treespec serialization hits a deprecated isinstance
        # check under Python 3.14 -- an upstream torch/Python-version mismatch in
        # production code this fixture calls for real, not something this repo triggers or
        # can fix. Scoped to this one call so any other FutureWarning still fails the suite.
        warnings.filterwarnings('ignore', message=r'.*isinstance\(treespec, LeafSpec\).*',
                                category=FutureWarning)
        export_pt2.worker_convert('test_vit4', pt2_path, False, 160, MODELS_SELECTED)
    return pt2_path


@pytest.fixture(scope='module')
def tiny_images(tmp_path_factory):
    from PIL import Image
    images_dir = tmp_path_factory.mktemp('images')
    for name in ['a.jpg', 'b.jpg']:
        Image.new('RGB', (160, 160), color=(128, 64, 32)).save(images_dir / name)
    zip_path = tmp_path_factory.mktemp('images_zip') / 'images.zip'
    with zipfile.ZipFile(zip_path, 'w') as z:
        for name in ['a.jpg', 'b.jpg']:
            z.write(images_dir / name, f'images/{name}')
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return str(images_dir), str(zip_path), sha256


def test_worker_convert_produces_a_valid_pt2(tiny_pt2):
    assert os.path.exists(tiny_pt2)
    with open(tiny_pt2, 'rb') as f:
        assert f.read(4) == b'PK\x03\x04'


def test_worker_pack_round_trip(tmp_path, tiny_pt2, tiny_images, capsys):
    images_dir, images_archive, images_sha256 = tiny_images
    output = str(tmp_path / 'test_vit4.zip')
    export_pt2.worker_pack('test_vit4', tiny_pt2, images_dir, images_archive, images_sha256,
                           output, 160, MODELS_SELECTED)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'ok', result
    assert result['images'] == 2

    with zipfile.ZipFile(output) as z:
        names = set(z.namelist())
        assert names == {'test_vit4.pt2', 'preprocessing.json', 'expected.json', 'contract.json',
                         'inputs.pt', 'outputs.pt'}
        expected = json.loads(z.read('expected.json'))
        assert set(expected) == {'a.jpg', 'b.jpg'}
        for entry in expected.values():
            assert len(entry['top5']) == 5
            assert len(set(entry['top5'])) == 5  # deterministic tie-break -> no duplicate ranks

        contract = json.loads(z.read('contract.json'))
        assert contract['inputs']['count'] == 2
        assert contract['inputs']['keys'] == ['a.jpg', 'b.jpg']
        assert contract['outputs']['keys_order'] == 'identical to inputs'
        assert contract['images_asset_sha256'] == images_sha256
        # The embedded .pt2 must be exactly the validated/loaded bytes, not a fresh disk read.
        with open(tiny_pt2, 'rb') as f:
            assert z.read('test_vit4.pt2') == f.read()


def test_worker_pack_rejects_images_mismatch(tmp_path, tiny_pt2, tiny_images, capsys):
    images_dir, images_archive, images_sha256 = tiny_images
    other_images_dir = tmp_path / 'other_images'
    other_images_dir.mkdir()
    from PIL import Image
    Image.new('RGB', (160, 160)).save(other_images_dir / 'different.jpg')

    output = str(tmp_path / 'mismatch.zip')
    export_pt2.worker_pack('test_vit4', tiny_pt2, str(other_images_dir), images_archive,
                           images_sha256, output, 160, MODELS_SELECTED)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'failed'
    assert 'sample mismatch' in result['error']
    assert not os.path.exists(output)


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_worker_pack_rejects_bad_model_archive_before_torch_export_load(tmp_path, tiny_images, monkeypatch, capsys):
    # zipfile.writestr itself warns on the second write of a name this test writes on
    # purpose, to build the fixture the assertion below actually exercises.
    images_dir, images_archive, images_sha256 = tiny_images
    bad_model = tmp_path / 'bad.pt2'
    with zipfile.ZipFile(bad_model, 'w') as z:
        z.writestr('models/model.json', b'{}')
        z.writestr('models/model.json', b'{}')  # duplicate member

    import torch
    called = []
    monkeypatch.setattr(torch.export, 'load', lambda *a, **kw: called.append(1))

    output = str(tmp_path / 'bad_out.zip')
    export_pt2.worker_pack('test_vit4', str(bad_model), images_dir, images_archive,
                           images_sha256, output, 160, MODELS_SELECTED)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'failed'
    assert not called, 'torch.export.load must never run once safe_zip_open rejects the archive'
    assert not os.path.exists(output)


# ---------------------------------------------------------------------------- worker_aoti_attempt


@pytest.mark.filterwarnings(
    r"ignore:`torch\.jit\.script_method` is:DeprecationWarning")
@pytest.mark.filterwarnings(
    r"ignore:.*isinstance\(treespec, LeafSpec\).*:FutureWarning")
def test_worker_aoti_attempt_matches_interpreter_output(tmp_path, tiny_pt2, tiny_images, capsys):
    images_dir, images_archive, images_sha256 = tiny_images
    release_zip = str(tmp_path / 'test_vit4.zip')
    export_pt2.worker_pack('test_vit4', tiny_pt2, images_dir, images_archive, images_sha256,
                           release_zip, 160, MODELS_SELECTED)
    capsys.readouterr()  # discard worker_pack's own result line

    export_pt2.worker_aoti_attempt('test_vit4', tiny_pt2, release_zip, MODELS_SELECTED)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'ok', result
    assert 'torch_version' in result['environment']
    assert 'error' not in result or result.get('error') is None


@pytest.mark.filterwarnings(
    r"ignore:`torch\.jit\.script_method` is:DeprecationWarning")
@pytest.mark.filterwarnings(
    r"ignore:.*isinstance\(treespec, LeafSpec\).*:FutureWarning")
def test_worker_aoti_attempt_records_failure_without_raising(tmp_path, tiny_pt2, tiny_images,
                                                              monkeypatch, capsys):
    images_dir, images_archive, images_sha256 = tiny_images
    release_zip = str(tmp_path / 'test_vit4.zip')
    export_pt2.worker_pack('test_vit4', tiny_pt2, images_dir, images_archive, images_sha256,
                           release_zip, 160, MODELS_SELECTED)
    capsys.readouterr()

    import torch._inductor
    monkeypatch.setattr(torch._inductor, 'aoti_compile_and_package',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('simulated compile failure')))
    export_pt2.worker_aoti_attempt('test_vit4', tiny_pt2, release_zip, MODELS_SELECTED)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['status'] == 'failed'
    assert 'simulated compile failure' in result['error']
