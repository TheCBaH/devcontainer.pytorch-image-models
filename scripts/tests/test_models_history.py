import subprocess

import pytest

import models_history


def _git(args, cwd):
    subprocess.run(['git'] + args, cwd=cwd, check=True, capture_output=True)


def _commit_manifest(repo, release_names, message):
    lines = ['models:']
    for name in release_names:
        lines.append(f'  {name}:\n    release: true')
    (repo / 'models-selected.yaml').write_text('\n'.join(lines) + '\n')
    _git(['add', 'models-selected.yaml'], repo)
    _git(['commit', '-m', message], repo)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(['init', '-q'], repo)
    _git(['config', 'user.email', 'test@example.invalid'], repo)
    _git(['config', 'user.name', 'Test'], repo)
    return repo


def test_no_previous_tag_is_first_release(repo):
    _commit_manifest(repo, ['a', 'b'], 'initial')
    tag, problems = models_history.unrecorded_removals('HEAD', str(repo / 'models-history.yaml'),
                                                       cwd=str(repo))
    assert tag is None
    assert problems == []


def test_removal_since_previous_tag_is_flagged(repo):
    _commit_manifest(repo, ['a', 'b', 'c'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _commit_manifest(repo, ['a', 'b'], 'v2')  # c removed
    tag, problems = models_history.unrecorded_removals('HEAD', str(repo / 'models-history.yaml'),
                                                       cwd=str(repo))
    assert tag == 'v0.0.1'
    assert problems == ['c']


def test_removal_recorded_in_history_is_not_flagged(repo, tmp_path):
    _commit_manifest(repo, ['a', 'b', 'c'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _commit_manifest(repo, ['a', 'b'], 'v2')
    history_path = tmp_path / 'models-history.yaml'
    history_path.write_text('models:\n  c:\n    removed_in: v0.0.2\n    reason: dropped\n    migrate_to: null\n')
    tag, problems = models_history.unrecorded_removals('HEAD', str(history_path), cwd=str(repo))
    assert tag == 'v0.0.1'
    assert problems == []


def test_model_that_returns_needs_no_entry(repo):
    _commit_manifest(repo, ['a', 'b', 'c'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _commit_manifest(repo, ['a', 'b'], 'v2')  # c removed
    _git(['tag', 'v0.0.2'], repo)
    _commit_manifest(repo, ['a', 'b', 'c'], 'v3')  # c returns
    tag, problems = models_history.unrecorded_removals('HEAD', str(repo / 'models-history.yaml'),
                                                       cwd=str(repo))
    # Compared only against the immediately preceding tag (v0.0.2), where c is already absent
    # on both sides -- nothing "removed since v0.0.2" at HEAD.
    assert tag == 'v0.0.2'
    assert problems == []


def test_base_pointing_at_the_about_to_be_published_tag_walks_to_its_parent(repo):
    _commit_manifest(repo, ['a', 'b', 'c'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _commit_manifest(repo, ['a', 'b'], 'v2')  # c removed
    _git(['tag', 'v0.0.2'], repo)
    tag, problems = models_history.unrecorded_removals('v0.0.2', str(repo / 'models-history.yaml'),
                                                       cwd=str(repo))
    assert tag == 'v0.0.1'
    assert problems == ['c']


def test_ambiguous_tags_at_the_same_commit_fail_loudly(repo):
    _commit_manifest(repo, ['a'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _git(['tag', 'v0.0.1-again'], repo)
    _commit_manifest(repo, ['a', 'b'], 'v2')
    with pytest.raises(SystemExit, match='multiple v\\* tags'):
        models_history.unrecorded_removals('HEAD', str(repo / 'models-history.yaml'), cwd=str(repo))


def test_shallow_checkout_fails_loudly(repo, tmp_path):
    _commit_manifest(repo, ['a', 'b'], 'v1')
    _git(['tag', 'v0.0.1'], repo)
    _commit_manifest(repo, ['a'], 'v2')
    shallow = tmp_path / 'shallow'
    # --depth is silently ignored for a plain local-path clone; file:// forces the real
    # smart-transport path, which actually honors it.
    subprocess.run(['git', 'clone', '--depth', '1', f'file://{repo}', str(shallow)],
                   check=True, capture_output=True)
    with pytest.raises(SystemExit, match='shallow'):
        models_history.unrecorded_removals('HEAD', str(shallow / 'models-history.yaml'),
                                           cwd=str(shallow))


def test_first_parent_ignores_a_merged_side_branch_tag(repo):
    _commit_manifest(repo, ['a', 'b', 'c'], 'main-1')
    _git(['tag', 'v0.0.1'], repo)
    _git(['checkout', '-b', 'side'], repo)
    _commit_manifest(repo, ['a', 'b', 'c', 'side-model'], 'side-1')
    _git(['tag', 'v0.0.2-side'], repo)
    _git(['checkout', 'master'], repo) if _default_branch_is_master(repo) else _git(['checkout', 'main'], repo)
    _git(['merge', '--no-ff', '-X', 'ours', '-m', 'merge side', 'side'], repo)
    _commit_manifest(repo, ['a', 'b'], 'main-2')  # c removed, on the mainline

    tag, problems = models_history.unrecorded_removals('HEAD', str(repo / 'models-history.yaml'),
                                                       cwd=str(repo))
    # The merged branch's v0.0.2-side tag must never be picked as "previous" -- only v0.0.1,
    # the true first-parent predecessor, and only the mainline's own removal (c) is flagged.
    assert tag == 'v0.0.1'
    assert problems == ['c']


def _default_branch_is_master(repo):
    result = subprocess.run(['git', 'branch', '--list', 'master'], cwd=repo,
                            capture_output=True, text=True)
    return bool(result.stdout.strip())
