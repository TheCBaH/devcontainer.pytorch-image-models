"""Render manifest.json/catalogue.json and the checksums.txt/release_assets() publish gate.

manifest.json hashes only the payload assets it inherently describes per model (images.zip,
each model .zip) -- never its own bytes or the other metadata files'. checksums.txt is
generated last, after manifest.json/catalogue.json/compat-report.json are all finalized,
listing every *other* uploaded asset; being a checksum file, it does not and cannot list
itself -- its own integrity comes from a separate attestation step over it directly, and from
release_assets()'s own two-pass, pinned-digest contract below.
"""
import hashlib
import os
import subprocess

from . import opgraph
from .archive import release_pt2_profile, safe_zip_open

CHECKSUMS_FILENAME = 'checksums.txt'


def sha256_file(path):
    """sha256 of a file's bytes, streamed rather than read whole -- a release .zip can be
    close to the release weight cap in size."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_asset_entry(path, url):
    """{'url', 'sha256', 'bytes'} for one uploaded asset -- the shape images/archive entries
    share in manifest.json."""
    return {'url': url, 'sha256': sha256_file(path), 'bytes': os.path.getsize(path)}


def archive_members(path, models_selected_path):
    """The member list of a just-built release .zip, read through the same bounded,
    duplicate-checked safe_zip_open every other .pt2/release-archive boundary uses -- this is
    trusted, locally-produced output, but there is no reason to read it any other way than
    everything else in this pipeline does.
    """
    with safe_zip_open(path, profile=release_pt2_profile(models_selected_path)) as z:
        return sorted(z.names())


def diff_retirements(release_names, history):
    """{name: {removed_in, reason, migrate_to}} for manifest.json's `retired` section.

    `history` is models-history.yaml's accumulated `models` ledger. A name currently back in
    the release tier is excluded regardless of history -- the selected/release pool reshuffles
    release over release as part of its coverage-optimization search (confirmed real behavior:
    starnet_s050 and test_convnext2 both cycled out and later back in), and a model that has
    returned is not retired, whatever it says here from an earlier gap.
    """
    return {name: entry for name, entry in history.items() if name not in release_names}


def commit_timestamp(commit, cwd=None):
    """The commit's own committer timestamp (`git show -s --format=%cI`) -- deterministic
    provenance instead of a wall-clock `generated_at` that would make two regenerations of the
    same commit's metadata incomparable."""
    return subprocess.run(
        ['git', 'show', '-s', '--format=%cI', commit], cwd=cwd,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def build_producer(repo, tag, commit, timm_version, torch_version, cwd=None):
    return {
        'repo': repo,
        'tag': tag,
        'commit': commit,
        'commit_timestamp': commit_timestamp(commit, cwd=cwd),
        'timm_version': timm_version,
        'torch_version': torch_version,
    }


def render_manifest(*, producer, selected_names, release_names, images_entry, model_entries,
                     retired):
    """`model_entries`: {name: {'url', 'sha256', 'bytes', 'members', optionally 'roles'}}."""
    models = {}
    for name, entry in sorted(model_entries.items()):
        archive = {k: entry[k] for k in ('url', 'sha256', 'bytes', 'members')}
        rendered = {'status': 'current', 'archive': archive}
        if entry.get('roles'):
            rendered['roles'] = sorted(entry['roles'])
        models[name] = rendered

    return {
        'schema_version': 1,
        'scope': 'full',
        'producer': producer,
        'selected_models': {'count': len(selected_names), 'names': sorted(selected_names)},
        'release_models': {'count': len(release_names), 'names': sorted(release_names)},
        'images': images_entry,
        'models': models,
        'retired': {
            name: {'removed_in': entry['removed_in'], 'reason': entry['reason'],
                   'migrate_to': entry.get('migrate_to')}
            for name, entry in sorted(retired.items())
        },
    }


def render_catalogue(*, selected_names, default_model, repo, commit, display_names=None,
                     aliases=None, deprecated=None, roles=None):
    """`deprecated`: {name: migrate_to_or_None} for a catalogue entry that is itself a
    superseded alias (distinct from `retired`, above -- a deprecated catalogue entry still
    resolves to a real, currently-published graph)."""
    display_names = display_names or {}
    aliases = aliases or {}
    deprecated = deprecated or {}
    roles = roles or {}

    models = {}
    for name in sorted(selected_names):
        is_deprecated = name in deprecated
        entry = {
            'display_name': display_names.get(name, name),
            'aliases': sorted(aliases.get(name, [])),
            'deprecated': is_deprecated,
            # Commit-SHA-addressed: a commit is genuinely immutable in git, unlike a tag --
            # the only URL in either document allowed to be called immutable.
            'graph_json_url': f'https://raw.githubusercontent.com/{repo}/{commit}/models/{name}/models/model.json',
        }
        if is_deprecated:
            entry['deprecated_migrate_to'] = deprecated[name]
        if roles.get(name):
            entry['roles'] = sorted(roles[name])
        models[name] = entry

    return {
        'schema_version': 1,
        'scope': 'full',
        'default_model': default_model,
        'models': models,
    }


def render_checksums(release_dir, relative_paths):
    """SHA256SUMS-style content ("<sha256>  <relative path>" per line, sorted), for every
    *other* uploaded asset under release_dir. Never includes checksums.txt itself -- it is
    generated last, after every asset it lists already exists."""
    lines = [f'{sha256_file(os.path.join(release_dir, rel))}  {rel}'
             for rel in sorted(relative_paths)]
    return '\n'.join(lines) + '\n'


def _parse_checksums(text):
    entries = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, sep, rel = line.partition('  ')
        if not sep or not digest or not rel:
            raise ValueError(f'malformed {CHECKSUMS_FILENAME} line: {line!r}')
        entries[rel] = digest
    return entries


def _verify_manifest_digests(manifest, checksums_entries):
    """The final publish-gate check: manifest.json's own recorded per-asset digests must
    still match the bytes release_assets() just rehashed -- catching a coordinated tamper
    that updates an asset and its checksums.txt line together but leaves manifest.json's
    entry for it stale.
    """
    problems = []
    images = manifest.get('images') or {}
    if images and checksums_entries.get('images.zip') != images.get('sha256'):
        problems.append(f'images.zip: manifest.json sha256 {images.get("sha256")!r} != '
                        f'verified digest {checksums_entries.get("images.zip")!r}')
    for name, entry in sorted((manifest.get('models') or {}).items()):
        archive = entry.get('archive') or {}
        rel = f'{name}.zip'
        if checksums_entries.get(rel) != archive.get('sha256'):
            problems.append(f'{name}: manifest.json archive.sha256 {archive.get("sha256")!r} '
                            f'!= verified digest {checksums_entries.get(rel)!r}')
    if problems:
        raise ValueError('manifest.json disagrees with the freshly verified checksums:\n  '
                         + '\n  '.join(problems))


def release_assets(release_dir, expect_checksums_sha256=None):
    """Recompute and re-verify the release asset list from disk every time this runs -- never
    trusts an in-memory list carried across a separate CI process, and never trusts an
    earlier verification.

    Always: reads checksums.txt, rehashes every listed file against release_dir, fails on any
    digest mismatch or missing file, and fails if release_dir contains any file not listed in
    checksums.txt.

    With `expect_checksums_sha256` (the second, pinned invocation, run immediately before
    `gh release create`): before rehashing anything, hashes checksums.txt itself and hard-fails
    if it does not match that pinned digest -- ruling out a coordinated edit to an asset *and*
    its checksums.txt line, made consistently, between the two invocations. Only after that pin
    succeeds does it rehash every asset and additionally re-verify manifest.json's own recorded
    digests against those freshly-computed values.

    Returns (checksums_sha256, [absolute path, ...], checksums.txt first).
    """
    checksums_path = os.path.join(release_dir, CHECKSUMS_FILENAME)
    if not os.path.exists(checksums_path):
        raise FileNotFoundError(f'{checksums_path}: not found')

    checksums_sha256 = sha256_file(checksums_path)
    if expect_checksums_sha256 is not None and checksums_sha256 != expect_checksums_sha256:
        raise ValueError(
            f'{CHECKSUMS_FILENAME} sha256 {checksums_sha256} does not match the pinned digest '
            f'{expect_checksums_sha256} from the first release_assets() invocation -- refusing '
            'to proceed: checksums.txt has changed since it was attested')

    with open(checksums_path) as f:
        entries = _parse_checksums(f.read())

    for rel, expected_digest in sorted(entries.items()):
        path = os.path.join(release_dir, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{rel}: listed in {CHECKSUMS_FILENAME} but missing from '
                                    f'{release_dir}')
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(f'{rel}: sha256 mismatch ({CHECKSUMS_FILENAME} says '
                             f'{expected_digest}, actual is {actual_digest})')

    listed = set(entries) | {CHECKSUMS_FILENAME}
    on_disk = set()
    for root, _dirs, files in os.walk(release_dir):
        for fname in files:
            on_disk.add(os.path.relpath(os.path.join(root, fname), release_dir))
    extra = on_disk - listed
    if extra:
        raise ValueError(f'{len(extra)} file(s) under {release_dir} are not listed in '
                         f'{CHECKSUMS_FILENAME}: {sorted(extra)}')

    if expect_checksums_sha256 is not None:
        manifest_path = os.path.join(release_dir, 'manifest.json')
        with open(manifest_path, 'rb') as f:
            manifest = opgraph.strict_json_loads(f.read())
        _verify_manifest_digests(manifest, entries)

    ordered = [os.path.join(release_dir, CHECKSUMS_FILENAME)] + \
              [os.path.join(release_dir, rel) for rel in sorted(entries)]
    return checksums_sha256, ordered
