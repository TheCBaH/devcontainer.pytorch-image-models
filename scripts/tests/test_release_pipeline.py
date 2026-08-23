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
    for n in ['a.jpg', 'b.jpg']:
        Image.new('RGB', (160, 160), color=(10, 20, 30)).save(images_dir / n)
    images_archive = release_dir / 'images.zip'
    with zipfile.ZipFile(images_archive, 'w') as z:
        for n in ['a.jpg', 'b.jpg']:
            z.write(images_dir / n, f'images/{n}')

    hf_home = str(root / '.hf-cache')

    build_args = _Args(manifest=str(manifest_path), models_dir=str(models_dir),
                       build_dir=str(build_dir), max_res=160, workers=1, timeout=300,
                       limit=None, keep_pt2=False, hf_home=hf_home)
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
