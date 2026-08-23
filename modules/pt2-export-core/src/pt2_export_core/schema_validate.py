"""Validate a document against one of this pipeline's committed JSON Schemas.

Every document this plan introduces (`op_facts.json`, `contract.json`, `manifest.json`,
`catalogue.json`, `compat-report.json`) has a schema committed in the caller's `schemas/`
directory, one file per document type plus any shared `$defs` they reference by relative
`$ref`. This module stays path-agnostic about where that directory lives -- `pt2_export_core`
has no opinion about which repo checkout it runs in -- so every call site passes its own
`schemas_dir` explicitly.
"""
import functools
import json
import os

from jsonschema import validators
from referencing import Registry, Resource


@functools.lru_cache(maxsize=None)
def _registry(schemas_dir):
    """One `Registry` per `schemas_dir`, holding every `*.schema.json` file in it, indexed
    both by its own `$id` (so relative `$ref`s between schemas in the same directory
    resolve) and by its bare filename (so a caller can also look one up by name alone).
    """
    resources = []
    for name in sorted(os.listdir(schemas_dir)):
        if not name.endswith('.schema.json'):
            continue
        with open(os.path.join(schemas_dir, name)) as f:
            contents = json.load(f)
        resource = Resource.from_contents(contents)
        resources.append((contents['$id'], resource))
        resources.append((name, resource))
    return Registry().with_resources(resources)


def load_schema(schema_name, schemas_dir):
    with open(os.path.join(schemas_dir, f'{schema_name}.schema.json')) as f:
        return json.load(f)


def validate_document(document, schema_name, schemas_dir):
    """Raise `jsonschema.exceptions.ValidationError` if `document` does not conform to
    `<schemas_dir>/<schema_name>.schema.json`.

    `document` must already be parsed data -- the caller reads it with
    `opgraph.strict_json_loads`, never a permissive `json.loads`, before this ever sees it,
    so a non-finite constant or a duplicate key fails at the parse boundary rather than
    silently passing schema validation on whichever value the permissive parser kept.
    """
    schema = load_schema(schema_name, schemas_dir)
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator_cls(schema, registry=_registry(schemas_dir)).validate(document)
