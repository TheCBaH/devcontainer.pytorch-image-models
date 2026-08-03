"""Fetch per-repo download counts from the HuggingFace Hub, as a popularity signal for
`pt2_export_core.selection`.

A report says what a model costs to export; it does not say whether anyone actually uses it.
An org/user's repos on the Hub carry a download count per pretrained tag, which is the
cheapest real usage signal available for a whole zoo at once.
"""
from huggingface_hub import HfApi


def fetch(author):
    """{repo name relative to `author`: downloads} for every repo under `author`.

    The `<author>/` prefix is stripped rather than kept as part of the key: a caller only
    ever fetches one author at a time, so every key would carry the same redundant prefix.
    """
    api = HfApi()
    prefix = f'{author}/'
    return {info.id.removeprefix(prefix): int(info.downloads or 0)
            for info in api.list_models(author=author, limit=None)}
