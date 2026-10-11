"""End-to-end integration test for the full release DAG: release -> manifest -> verify-release
-> release-assets, against one real, tiny, genuinely-pretrained release-tier model
(test_efficientnet_gn, ~1.4MB, real hosted weights) -- exercising actual network downloads and
real subprocess workers, not fabricated fixtures.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import export_pt2

NAME = 'test_efficientnet_gn'


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _repo_head_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=export_pt2.REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope='module')
def release_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp('release_pipeline')
    manifest_path = root / 'models-selected.yaml'
    manifest_path.write_text(
        'selection:\n'
        '  max_weight_mb: 150.0\n'
        '  release_max_weight_mb: 100.0\n'
        'models:\n'
        f'  {NAME}:\n'
        '    release: true\n'
    )
    models_dir = root / 'models'
    build_dir = root / '.build-scratch'
    release_dir = root / 'release'
    work_dir = root / 'work'
    images_dir = root / 'images'
    images_dir.mkdir()
    release_dir.mkdir()

    from PIL import Image
    from test_input_contracts import patterned
    for n, dims in [('a.jpg', (181, 239)), ('b.jpg', (243, 179))]:
        patterned(*dims).save(images_dir / n)
    images_archive = release_dir / 'images.zip'
    with zipfile.ZipFile(images_archive, 'w') as z:
        for n in ['a.jpg', 'b.jpg']:
            z.write(images_dir / n, f'images/{n}')

    hf_home = str(root / '.hf-cache')

    build_args = _Args(manifest=str(manifest_path), models_dir=str(models_dir),
                       build_dir=str(build_dir), max_res=160, workers=1, timeout=300,
                       limit=None, keep_pt2=False, hf_home=hf_home, policy='fp32', dtype=None)
    assert export_pt2.cmd_build(build_args) == 0

    # cmd_fetch downloads for real, in-process (timm.create_model(pretrained=True) directly,
    # not through a subprocess worker), which needs HF_HOME set *before* huggingface_hub's
    # first import -- true for a real one-shot CLI process (main() sets it before any lazy
    # `import timm`), but not reliably true inside a shared, long-lived pytest process where
    # an earlier test module may already have imported huggingface_hub with a different
    # HF_HOME baked into its module-level constants. Running fetch as a real subprocess (like
    # `make models.fetch` does) sidesteps that entirely -- a fresh process, fresh imports.
    subprocess.run(
        [sys.executable, os.path.join(export_pt2.REPO_ROOT, 'scripts', 'export_pt2.py'),
         '--manifest', str(manifest_path), '--hf-home', hf_home, 'fetch', '--only', NAME],
        check=True, capture_output=True, text=True,
    )

    release_args = _Args(manifest=str(manifest_path), images=str(images_dir),
                         images_archive=str(images_archive), release_dir=str(release_dir),
                         work_dir=str(work_dir), workers=1, timeout=300, only=None,
                         max_res=160, hf_home=hf_home, skip_aoti=False, aoti_timeout=180)
    assert export_pt2.cmd_release(release_args) == 0

    return {
        'manifest': str(manifest_path), 'models_dir': str(models_dir),
        'release_dir': str(release_dir), 'work_dir': str(work_dir),
    }


def test_release_produces_archive_and_pack_results(release_pipeline):
    assert os.path.exists(os.path.join(release_pipeline['release_dir'], f'{NAME}.zip'))
    pack_results_path = os.path.join(release_pipeline['work_dir'], export_pt2.PACK_RESULTS_FILENAME)
    with open(pack_results_path) as f:
        state = json.load(f)
    assert state['scope'] == 'full'
    assert state['results'][NAME]['status'] == 'ok'
    # A real AOTInductor-CPU compile+load+run, isolated in its own subprocess, ran and
    # matched the interpreter's own output for a real sample.
    assert state['results'][NAME]['aoti']['status'] == 'ok', state['results'][NAME]['aoti']
    assert 'torch_version' in state['results'][NAME]['aoti']['environment']


def test_manifest_generates_all_three_documents(release_pipeline, tmp_path):
    roles_path = tmp_path / 'models-roles.yaml'
    roles_path.write_text(f'models:\n  {NAME}:\n    - test_fixture_role\n')

    manifest_args = _Args(
        manifest=release_pipeline['manifest'], models_dir=release_pipeline['models_dir'],
        release_dir=release_pipeline['release_dir'], work_dir=release_pipeline['work_dir'],
        images_archive=None, history=os.path.join(export_pt2.REPO_ROOT, 'models-history.yaml'),
        roles=str(roles_path),
        repo='example-org/example-repo', tag='v9.9.9', commit=_repo_head_commit(),
        default_model=NAME, timm_version=None, torch_version=None,
    )
    assert export_pt2.cmd_manifest(manifest_args) == 0

    release_dir = release_pipeline['release_dir']
    with open(os.path.join(release_dir, 'manifest.json')) as f:
        manifest_doc = json.load(f)
    with open(os.path.join(release_dir, 'catalogue.json')) as f:
        catalogue_doc = json.load(f)
    with open(os.path.join(release_dir, 'compat-report.json')) as f:
        compat_doc = json.load(f)

    assert manifest_doc['scope'] == 'full'
    assert NAME in manifest_doc['models']
    assert manifest_doc['models'][NAME]['archive']['members']  # non-empty member list
    assert manifest_doc['models'][NAME]['roles'] == ['test_fixture_role']
    assert catalogue_doc['models'][NAME]['roles'] == ['test_fixture_role']
    assert catalogue_doc['default_model'] == NAME
    assert compat_doc['models'][NAME]['classification'] == 'runnable'
    assert compat_doc['models'][NAME]['backends']['interpreter']['result'] == 'passed'
    assert compat_doc['models'][NAME]['backends']['aot_inductor_cpu']['result'] == 'passed'


def test_verify_release_passes_and_writes_checksums(release_pipeline):
    verify_args = _Args(manifest=release_pipeline['manifest'],
                        models_dir=release_pipeline['models_dir'],
                        release_dir=release_pipeline['release_dir'])
    assert export_pt2.cmd_verify_release(verify_args) == 0
    checksums_path = os.path.join(release_pipeline['release_dir'], 'checksums.txt')
    assert os.path.exists(checksums_path)
    with open(checksums_path) as f:
        text = f.read()
    assert f'{NAME}.zip' in text
    assert 'manifest.json' in text
    assert 'checksums.txt' not in text  # never lists itself


def test_verify_release_catches_a_tampered_archive(release_pipeline, tmp_path):
    import shutil
    tampered_dir = tmp_path / 'tampered_release'
    shutil.copytree(release_pipeline['release_dir'], tampered_dir)
    with open(os.path.join(tampered_dir, f'{NAME}.zip'), 'r+b') as f:
        f.seek(100)
        f.write(b'\x00' * 4)  # flip a few bytes inside the archive

    verify_args = _Args(manifest=release_pipeline['manifest'],
                        models_dir=release_pipeline['models_dir'], release_dir=str(tampered_dir))
    assert export_pt2.cmd_verify_release(verify_args) == 1


def test_release_assets_two_pass_pinned_flow(release_pipeline):
    first_args = _Args(release_dir=release_pipeline['release_dir'], expect_checksums_sha256=None)
    assert export_pt2.cmd_release_assets(first_args) == 0

    checksums_sha256, _ = export_pt2.manifest_mod.release_assets(release_pipeline['release_dir'])
    second_args = _Args(release_dir=release_pipeline['release_dir'],
                        expect_checksums_sha256=checksums_sha256)
    assert export_pt2.cmd_release_assets(second_args) == 0

    wrong_pin_args = _Args(release_dir=release_pipeline['release_dir'],
                           expect_checksums_sha256='0' * 64)
    with pytest.raises(ValueError, match='does not match the pinned digest'):
        export_pt2.cmd_release_assets(wrong_pin_args)


def test_evaluation_and_input_assets_share_bytes_and_legacy_replays(release_pipeline):
    import io
    import torch
    directory = release_pipeline['release_dir']
    with zipfile.ZipFile(os.path.join(directory, f'{NAME}.zip')) as full, \
         zipfile.ZipFile(os.path.join(directory, f'{NAME}-eval.zip')) as evaluation:
        assert sorted(evaluation.namelist()) == export_pt2.input_assets.EVALUATION_MEMBERS
        for member in evaluation.namelist():
            assert evaluation.read(member) == full.read(member)
        standalone = Path(directory, f'{NAME}-inputs.pt').read_bytes()
        assert standalone == full.read('inputs.pt')
        # The v1 playback path ignores the additive JSON members.
        legacy = json.loads(full.read('contract.json'))
        assert legacy['schema_version'] == 1
        module = torch.export.load(io.BytesIO(full.read(f'{NAME}.pt2'))).module()
        inputs = torch.load(io.BytesIO(full.read('inputs.pt')), weights_only=True)
        outputs = torch.load(io.BytesIO(full.read('outputs.pt')), weights_only=True)
        with torch.no_grad():
            for key in inputs:
                assert export_pt2._numeric_match(module(inputs[key].unsqueeze(0))[0].float(),
                    outputs[key], legacy['expected']['atol'], legacy['expected']['rtol'])


def test_independent_reproduction_does_not_open_full_archive(release_pipeline, tmp_path):
    import shutil
    directory = tmp_path / 'inputs-only'
    shutil.copytree(release_pipeline['release_dir'], directory)
    (directory / f'{NAME}.zip').unlink()
    args = _Args(manifest=release_pipeline['manifest'], models_dir=release_pipeline['models_dir'],
                 release_dir=str(directory))
    assert export_pt2.cmd_verify_inputs(args) == 0


@pytest.mark.parametrize('mutation', ['missing', 'unexpected', 'input', 'evaluation',
                                     'unresolved', 'graph', 'descriptor', 'geometry', 'images'])
def test_v2_incomplete_or_inconsistent_release_never_gets_checksums(release_pipeline, tmp_path, mutation):
    import shutil
    from pathlib import Path
    directory = tmp_path / mutation
    shutil.copytree(release_pipeline['release_dir'], directory)
    index = json.loads((directory / 'manifest-v2.json').read_bytes())
    if mutation == 'missing': (directory / f'{NAME}-inputs.pt').unlink()
    if mutation == 'unexpected': (directory / 'unexpected.txt').write_text('extra')
    if mutation == 'input': (directory / f'{NAME}-inputs.pt').write_bytes(b'bad')
    if mutation == 'images': (directory / 'images.pt').write_bytes(b'bad')
    if mutation == 'unresolved': index['models'][NAME]['inputs'] = 'missing/inputs'
    if mutation in ('evaluation', 'graph', 'descriptor', 'geometry'):
        path = directory / f'{NAME}-eval.zip'
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        if mutation == 'evaluation': members['forbidden.pt2'] = b'bad'
        if mutation == 'graph':
            doc = json.loads(members['contract-v2.json'])
            doc['graph']['sha256'] = 'a' * 64
            members['contract-v2.json'] = export_pt2.input_assets.json_bytes(doc)
        if mutation in ('descriptor', 'geometry'):
            doc = json.loads(members['cases.json'])
            if mutation == 'descriptor':
                doc['cases'][0]['inputs']['image']['tensor']['sha256'] = 'a' * 64
            else:
                doc['cases'][0]['geometry'][0]['after'][0] += 1
            members['cases.json'] = export_pt2.input_assets.json_bytes(doc)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for n, data in members.items(): z.writestr(n, data)
        asset = index['assets'][f'{NAME}/evaluation']
        asset.update(sha256=export_pt2.manifest_mod.sha256_file(path), bytes=path.stat().st_size,
                     members=sorted(members))
    (directory / 'manifest-v2.json').write_bytes(export_pt2.input_assets.json_bytes(index))
    args = _Args(manifest=release_pipeline['manifest'], models_dir=release_pipeline['models_dir'],
                 release_dir=str(directory))
    assert export_pt2.cmd_verify_release(args) == 1
    assert not (directory / 'checksums.txt').exists()


@pytest.mark.parametrize('mutation', ['hash', 'size', 'duplicate', 'population', 'unresolved', 'extra', 'members'])
def test_first_and_second_publication_pass_reject_invalid_v2_index(release_pipeline, tmp_path, mutation):
    import shutil
    directory = tmp_path / mutation
    shutil.copytree(release_pipeline['release_dir'], directory)
    path = directory / 'manifest-v2.json'
    doc = json.loads(path.read_bytes())
    asset = doc['assets'][f'{NAME}/inputs']
    if mutation == 'hash': asset['sha256'] = 'b' * 64
    if mutation == 'size': asset['bytes'] += 1
    if mutation == 'duplicate': asset['filename'] = 'images.pt'
    if mutation == 'population': doc['release_models']['names'] = []
    if mutation == 'unresolved': doc['models'][NAME]['inputs'] = 'missing'
    if mutation == 'extra': doc['assets']['unexpected'] = dict(asset, filename='other.pt')
    if mutation == 'members': doc['assets'][f'{NAME}/evaluation']['members'] = ['bad']
    path.write_bytes(export_pt2.input_assets.json_bytes(doc))
    files = [p.name for p in directory.iterdir() if p.name != 'checksums.txt']
    (directory / 'checksums.txt').write_text(export_pt2.manifest_mod.render_checksums(str(directory), files))
    for pin in [None, export_pt2.manifest_mod.sha256_file(directory / 'checksums.txt')]:
        with pytest.raises(Exception):
            export_pt2.manifest_mod.release_assets(str(directory), expect_checksums_sha256=pin,
                                                    schemas_dir=export_pt2.SCHEMAS_DIR)
