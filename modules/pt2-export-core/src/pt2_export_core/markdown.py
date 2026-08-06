"""Markdown rendering helpers shared by the generated reports.

These files are read on GitHub, where they are long enough (a per-operator section for every
operator in the zoo, a table for every model family) that a reader arrives looking for one
entry rather than reading top to bottom. So each of them opens with an index table whose
first column links into the section below -- which means minting the same heading anchors
GitHub's renderer does.
"""
import re


def heading_anchor(text, seen):
    """The fragment GitHub gives a heading with this text, e.g. `add_.Tensor` -> `add_tensor`.

    Lowercased, punctuation dropped, spaces hyphenated, and repeats disambiguated with a
    `-1`/`-2` suffix -- so `seen` is a mutable {slug: count} the caller carries across every
    heading of one document, in the order the headings are emitted.
    """
    slug = re.sub(r'[^\w\- ]', '', text.lower()).replace(' ', '-')
    n = seen.get(slug, 0)
    seen[slug] = n + 1
    return slug if not n else f'{slug}-{n}'
