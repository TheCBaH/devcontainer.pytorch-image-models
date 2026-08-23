#!/usr/bin/env python3
"""Retirement checker: a release-tier model present in the previous release tag's
models-selected.yaml but absent from the current commit must be recorded in
models-history.yaml, or `check` fails.

Scope, deliberately narrower than "every removal, ever": only release-tier models (the ones
with a published archive a consumer could actually depend on) are tracked, and only against
the *immediately preceding* release tag, not full history. The selected pool (~100 models)
reshuffles a meaningful fraction release over release as part of its coverage-optimization
search -- confirmed real behavior between v0.0.1 and v0.0.3, where starnet_s050 and
test_convnext2 both dropped out of the release tier and later returned. Treating every
disappearance as a permanent retirement needing a migration entry would either fabricate false
"gone forever" records for models mid-reshuffle, or fail this check on completely ordinary
churn. Comparing only against the immediately preceding tag means a model that has already
returned by the commit being checked needs no entry at all.

Commands:
  check   fail if a release-tier model vanished since the previous release tag with no entry
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(args, cwd=None):
    return subprocess.run(['git'] + args, cwd=cwd or REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout


def _release_tags_at(commit, cwd=None):
    return [t for t in _git(['tag', '--points-at', commit], cwd=cwd).splitlines() if t.startswith('v')]


def previous_release_tag(base, cwd=None):
    """(tag, commit) of the nearest v* tag on `base`'s first-parent chain, *excluding* a tag
    that points at `base` itself (release.yml's HEAD is the tag about to be published -- the
    previous release is the one before it, not itself). None, None if there is no earlier
    release (first-release case).

    Walks strictly first-parent -- not `git describe`, which can surface a tag brought in by a
    merge rather than the true mainline predecessor. Requires full history (fetch-depth: 0);
    a shallow checkout fails loudly rather than silently reporting "no previous release".
    """
    is_shallow = _git(['rev-parse', '--is-shallow-repository'], cwd=cwd).strip() == 'true'
    if is_shallow:
        sys.exit('this is a shallow checkout -- models_history.py check requires full history '
                 '(actions/checkout with fetch-depth: 0)')

    walk_from = base
    if _release_tags_at(base, cwd=cwd):
        # `base` is itself the tag about to be published: walk from its parent so the
        # "previous release" search does not just find itself.
        walk_from = f'{base}^'

    try:
        commits = _git(['rev-list', '--first-parent', walk_from], cwd=cwd).splitlines()
    except subprocess.CalledProcessError as e:
        sys.exit(f'git rev-list --first-parent {walk_from} failed:\n{e.stderr}')
    if not commits:
        return None, None

    for commit in commits:
        tags = _release_tags_at(commit, cwd=cwd)
        if not tags:
            continue
        if len(tags) > 1:
            sys.exit(f'{commit}: multiple v* tags point at this commit '
                     f'({", ".join(sorted(tags))}) -- refusing to guess which is the previous '
                     'release')
        return tags[0], commit
    return None, None


def release_names_at(ref, cwd=None):
    try:
        content = _git(['show', f'{ref}:models-selected.yaml'], cwd=cwd)
    except subprocess.CalledProcessError:
        sys.exit(f'{ref}: models-selected.yaml not found at this ref')
    document = yaml.safe_load(content) or {}
    models = document.get('models') or {}
    return {name for name, entry in models.items() if entry.get('release')}


def load_history(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    return document.get('models') or {}


def unrecorded_removals(base, history_path, cwd=None):
    """(previous_tag_or_None, sorted [name, ...] removed since it with no models-history.yaml
    entry). An empty previous_tag means there is no earlier release to compare against."""
    tag, _commit = previous_release_tag(base, cwd=cwd)
    if tag is None:
        return None, []
    previous = release_names_at(tag, cwd=cwd)
    current = release_names_at(base, cwd=cwd)
    disappeared = previous - current
    history = load_history(history_path)
    return tag, sorted(name for name in disappeared if name not in history)


def cmd_check(args):
    tag, problems = unrecorded_removals(args.base, args.history)
    if tag is None:
        print('no previous release tag found on the first-parent chain -- nothing to check '
             '(first release)', file=sys.stderr)
        return 0
    if problems:
        print(f'{len(problems)} release-tier model(s) removed since {tag} with no '
             f'{os.path.basename(args.history)} entry:', file=sys.stderr)
        for name in problems:
            print(f'  {name}', file=sys.stderr)
        print(f'\nAdd an entry to {args.history} (removed_in, reason, migrate_to) for each, '
             'or confirm this is temporary reshuffling and not an actual retirement.',
             file=sys.stderr)
        return 1
    print(f'no unrecorded release-tier removals since {tag}', file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--history', default=os.path.join(REPO_ROOT, 'models-history.yaml'))
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('check', help='fail if a release-tier removal since the previous tag is unrecorded')
    p.add_argument('--base', default='HEAD', help='ref to check (default: HEAD)')
    p.set_defaults(func=cmd_check)

    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
