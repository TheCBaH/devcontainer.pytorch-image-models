#!/usr/bin/env bash
# Show what changed in the committed PT2 graphs, readably.
#
# The JSON in models/ is the serializer's own output: minified onto a single line. A plain
# `git diff` on it reports "one line changed" and nothing else, so both sides are pretty-
# printed with jq before diffing. Exits non-zero on any difference, which is what makes this
# usable as a CI check as well as a debugging aid.
set -euo pipefail

MODELS_DIR="${1:-models}"

changed="$(git diff --name-only HEAD -- "$MODELS_DIR")"
untracked="$(git ls-files --others --exclude-standard -- "$MODELS_DIR")"

if [ -z "$changed" ] && [ -z "$untracked" ]; then
    echo "All model JSON files match HEAD."
    exit 0
fi

rc=0
while IFS= read -r file; do
    [ -z "$file" ] && continue
    echo "=== $file ==="
    if [ -f "$file" ]; then
        diff -u \
            --label "HEAD:$file" \
            --label "tree:$file" \
            <(git show "HEAD:$file" | jq .) \
            <(jq . "$file") || rc=1
    else
        echo "(deleted from working tree)"
        rc=1
    fi
done <<< "$changed"

while IFS= read -r file; do
    [ -z "$file" ] && continue
    echo "=== $file ==="
    echo "(new file, not in HEAD)"
    rc=1
done <<< "$untracked"

if [ "$rc" -ne 0 ]; then
    echo ""
    echo "ERROR: models/ differs from HEAD. Run 'make models' and commit the result."
fi
exit "$rc"
