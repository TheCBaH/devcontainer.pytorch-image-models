import hashlib
import json
import os

import pytest

from pt2_export_core import manifest


def _write(path, data):
    with open(path, 'wb') as f:
        f.write(data)
    return path


def test_sha256_file_matches_hashlib(tmp_path):
    path = _write(str(tmp_path / 'a.bin'), b'hello world' * 1000)
    assert manifest.sha256_file(path) == hashlib.sha256(b'hello world' * 1000).hexdigest()


def test_build_asset_entry(tmp_path):
    path = _write(str(tmp_path / 'a.bin'), b'xyz')
    entry = manifest.build_asset_entry(path, 'https://example.invalid/a.bin')
    assert entry == {
        'url': 'https://example.invalid/a.bin',
        'sha256': hashlib.sha256(b'xyz').hexdigest(),
        'bytes': 3,
    }


def test_diff_retirements_excludes_names_currently_in_release_tier():
    history = {
        'gone': {'removed_in': 'v0.0.2', 'reason': 'weight cap', 'migrate_to': 'gone2'},
        'returned': {'removed_in': 'v0.0.2', 'reason': 'reshuffle', 'migrate_to': None},
    }
    retired = manifest.diff_retirements({'returned', 'other'}, history)
    assert retired == {'gone': history['gone']}


def test_render_manifest_shape():
    doc = manifest.render_manifest(
        producer={'repo': 'org/repo', 'tag': 'v0.0.4', 'commit': 'a' * 40,
                  'commit_timestamp': '2026-08-21T00:00:00+00:00',
                  'timm_version': '1.0', 'torch_version': '2.9.0'},
        selected_names={'m1', 'm2'},
        release_names={'m1'},
        images_entry={'url': 'https://x/images.zip', 'sha256': 'b' * 64, 'bytes': 10},
        model_entries={'m1': {'url': 'https://x/m1.zip', 'sha256': 'c' * 64, 'bytes': 20,
                              'members': ['m1.pt2', 'contract.json']}},
        retired={'gone': {'removed_in': 'v0.0.2', 'reason': 'r', 'migrate_to': None}},
    )
    assert doc['schema_version'] == 1
    assert doc['scope'] == 'full'
    assert doc['models']['m1']['status'] == 'current'
    assert doc['models']['m1']['archive']['members'] == ['m1.pt2', 'contract.json']
    assert doc['retired']['gone']['migrate_to'] is None
    assert doc['selected_models'] == {'count': 2, 'names': ['m1', 'm2']}


def test_render_manifest_validates_against_schema():
    from pt2_export_core.schema_validate import validate_document
    schemas_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'schemas'))
    doc = manifest.render_manifest(
        producer={'repo': 'org/repo', 'tag': 'v0.0.4', 'commit': 'a' * 40,
                  'commit_timestamp': '2026-08-21T00:00:00+00:00',
                  'timm_version': '1.0', 'torch_version': '2.9.0'},
        selected_names={'m1'},
        release_names={'m1'},
        images_entry={'url': 'https://x/images.zip', 'sha256': 'b' * 64, 'bytes': 10},
        model_entries={'m1': {'url': 'https://x/m1.zip', 'sha256': 'c' * 64, 'bytes': 20,
                              'members': ['m1.pt2']}},
        retired={},
    )
    validate_document(doc, 'manifest', schemas_dir)


def test_render_catalogue_deprecated_entry_requires_migrate_to():
    doc = manifest.render_catalogue(
        selected_names={'m1', 'old_name'}, default_model='m1', repo='org/repo', commit='a' * 40,
        deprecated={'old_name': 'm1'},
    )
    assert doc['models']['old_name']['deprecated'] is True
    assert doc['models']['old_name']['deprecated_migrate_to'] == 'm1'
    assert doc['models']['m1']['deprecated'] is False
    assert 'deprecated_migrate_to' not in doc['models']['m1']
    assert doc['models']['m1']['graph_json_url'] == \
        f'https://raw.githubusercontent.com/org/repo/{"a"*40}/models/m1/models/model.json'


def test_render_checksums_and_parse_round_trip(tmp_path):
    (tmp_path / 'a.zip').write_bytes(b'aaa')
    (tmp_path / 'b.zip').write_bytes(b'bb')
    text = manifest.render_checksums(str(tmp_path), ['a.zip', 'b.zip'])
    entries = manifest._parse_checksums(text)
    assert entries == {
        'a.zip': hashlib.sha256(b'aaa').hexdigest(),
        'b.zip': hashlib.sha256(b'bb').hexdigest(),
    }
    assert 'checksums.txt' not in text


# ---------------------------------------------------------------------------- release_assets


def _build_release_dir(tmp_path):
    release_dir = tmp_path / 'release'
    release_dir.mkdir()
    (release_dir / 'm1.zip').write_bytes(b'm1-bytes')
    (release_dir / 'images.zip').write_bytes(b'images-bytes')
    manifest_doc = {
        'images': {'sha256': hashlib.sha256(b'images-bytes').hexdigest()},
        'models': {'m1': {'archive': {'sha256': hashlib.sha256(b'm1-bytes').hexdigest()}}},
    }
    (release_dir / 'manifest.json').write_text(json.dumps(manifest_doc))
    checksums_text = manifest.render_checksums(str(release_dir), ['m1.zip', 'images.zip', 'manifest.json'])
    (release_dir / 'checksums.txt').write_text(checksums_text)
    return release_dir, manifest_doc


def test_release_assets_first_invocation(tmp_path):
    release_dir, _ = _build_release_dir(tmp_path)
    checksums_sha256, paths = manifest.release_assets(str(release_dir))
    assert os.path.basename(paths[0]) == 'checksums.txt'
    assert {os.path.basename(p) for p in paths} == {'checksums.txt', 'm1.zip', 'images.zip', 'manifest.json'}
    assert checksums_sha256 == manifest.sha256_file(str(release_dir / 'checksums.txt'))


def test_release_assets_rejects_extra_unlisted_file(tmp_path):
    release_dir, _ = _build_release_dir(tmp_path)
    (release_dir / 'stray.txt').write_bytes(b'not listed')
    with pytest.raises(ValueError, match='not listed'):
        manifest.release_assets(str(release_dir))


def test_release_assets_rejects_tampered_asset(tmp_path):
    release_dir, _ = _build_release_dir(tmp_path)
    (release_dir / 'm1.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256 mismatch'):
        manifest.release_assets(str(release_dir))


def test_release_assets_second_invocation_pins_checksums_digest(tmp_path):
    release_dir, _ = _build_release_dir(tmp_path)
    checksums_sha256, _ = manifest.release_assets(str(release_dir))
    # Passes when the pin matches.
    manifest.release_assets(str(release_dir), expect_checksums_sha256=checksums_sha256)
    # Fails when it doesn't -- even though checksums.txt is still internally self-consistent.
    with pytest.raises(ValueError, match='does not match the pinned digest'):
        manifest.release_assets(str(release_dir), expect_checksums_sha256='0' * 64)


def test_coordinated_tamper_between_invocations_fails_the_pin(tmp_path):
    """An asset AND its checksums.txt line are changed together, consistently, strictly
    between the two invocations -- the pin must still fail, since two independently-passing
    rehashes are not enough on their own (this is exactly what round 8 of the plan's review
    process found: the earlier "rehash twice" design did not actually rule this out)."""
    release_dir, _ = _build_release_dir(tmp_path)
    first_digest, _ = manifest.release_assets(str(release_dir))

    (release_dir / 'm1.zip').write_bytes(b'swapped-in-bytes')
    new_checksums = manifest.render_checksums(str(release_dir), ['m1.zip', 'images.zip', 'manifest.json'])
    (release_dir / 'checksums.txt').write_text(new_checksums)  # kept internally consistent

    with pytest.raises(ValueError, match='does not match the pinned digest'):
        manifest.release_assets(str(release_dir), expect_checksums_sha256=first_digest)


def test_manifest_json_stale_digest_caught_by_pinned_invocation(tmp_path):
    """checksums.txt and the pin both stay valid (the asset's checksums.txt line is updated to
    match), but manifest.json's own recorded digest for that asset is left stale -- the
    pinned invocation's manifest-rebinding check must catch this independently of the pin."""
    release_dir, manifest_doc = _build_release_dir(tmp_path)
    first_digest, _ = manifest.release_assets(str(release_dir))

    (release_dir / 'm1.zip').write_bytes(b'new-bytes-not-in-manifest')
    new_checksums = manifest.render_checksums(str(release_dir), ['m1.zip', 'images.zip', 'manifest.json'])
    (release_dir / 'checksums.txt').write_text(new_checksums)
    # Re-pin against the *new* checksums.txt digest (simulating a re-attestation) so the pin
    # itself passes -- only the manifest.json <-> asset binding is now stale.
    new_pin = manifest.sha256_file(str(release_dir / 'checksums.txt'))

    with pytest.raises(ValueError, match='manifest.json disagrees'):
        manifest.release_assets(str(release_dir), expect_checksums_sha256=new_pin)
