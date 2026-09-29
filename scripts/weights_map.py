#!/usr/bin/env python3
"""Map every committed graph's weights onto the HuggingFace safetensors checkpoint they come from.

The committed graphs carry no weight blobs, only `data/weights/model_weights_config.json`: an
index of every parameter and persistent buffer by its `state_dict` name, with shape and dtype.
timm publishes its pretrained weights on the Hub as `model.safetensors`, keyed by the same
names. `build` records that correspondence per model as `models/<name>/models/safetensors.json`:
where the checkpoint is (repo, pinned revision, URL, sha256), and which checkpoint tensor feeds
each graph weight. Constants (non-persistent buffers, lifted tensors) are mapped too where the
checkpoint happens to carry them, and listed under `unmapped` where it does not -- those are
computed by the model itself, so no checkpoint has them.

A model is mapped only if *every* graph weight has a checkpoint tensor of the same shape; a
dtype may differ only between floating types (the fp16/bf16 `cast` graphs). Models the manifest
gives a Hub source for but that cannot be mapped are pinned, with the reason, in
safetensors-unmapped.yaml, and `verify` fails both on a model missing from it and on one
listed there that is mapped after all.

Commands:
  build         (network) write safetensors.json for every graph, and safetensors-unmapped.yaml
  verify        (offline) re-check every committed safetensors.json against its graph
  check-values  (network, downloads) prove timm's own pretrained load equals the mapped tensors
"""
import argparse
import json
import os
import sys

import yaml

from pt2_export_core import opgraph, schema_validate
from pt2_export_core.archive import load_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMAS_DIR = os.path.join(REPO_ROOT, 'schemas')
WEIGHTS_MAP = os.path.join('models', 'safetensors.json')
FILENAME = 'model.safetensors'

# torch._export.serde.schema.ScalarType -> safetensors dtype
SCALAR_TYPES = {1: 'U8', 2: 'I8', 3: 'I16', 4: 'I32', 5: 'I64', 6: 'F16', 7: 'F32', 8: 'F64',
                12: 'BOOL', 13: 'BF16'}
FLOATING = {'F16', 'BF16', 'F32', 'F64'}


def _graph_tensors(model_dir, kind):
    path = os.path.join(model_dir, 'data', kind, f'model_{kind}_config.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        config = json.load(f)['config']
    return {name: ([s['as_int'] for s in entry['tensor_meta']['sizes']],
                   SCALAR_TYPES.get(entry['tensor_meta']['dtype'], str(entry['tensor_meta']['dtype'])))
            for name, entry in config.items()}


def _compatible(graph, checkpoint):
    """None if a checkpoint tensor can feed a graph tensor, else why not."""
    (shape, dtype), (ck_shape, ck_dtype) = graph, checkpoint
    if shape != ck_shape:
        return f'shape {ck_shape} != {shape}'
    if dtype != ck_dtype and not (dtype in FLOATING and ck_dtype in FLOATING):
        return f'dtype {ck_dtype} != {dtype}'
    return None


def map_tensors(weights, constants, header, prefix=''):
    """(tensors, unmapped) for one graph against one checkpoint header, or raise ValueError
    naming the first graph weight the checkpoint cannot supply.

    `weights`/`constants` are {name: (shape, dtype)} from the graph, `header` the same for the
    checkpoint. A graph name is `prefix` + checkpoint key: the autocast graphs wrap the timm
    model as `.model`. Every weight must map; a constant maps only if the checkpoint has it.
    """
    def key(name):
        return name[len(prefix):] if name.startswith(prefix) else None

    tensors, unmapped = {}, []
    for name, graph in sorted(weights.items()):
        if key(name) not in header:
            raise ValueError(f'{name}: not in checkpoint')
        problem = _compatible(graph, header[key(name)])
        if problem:
            raise ValueError(f'{name}: {problem}')
        tensors[name] = key(name)
    for name, graph in sorted(constants.items()):
        if key(name) in header and _compatible(graph, header[key(name)]) is None:
            tensors[name] = key(name)
        else:
            unmapped.append(name)
    return ({name: {'key': k, 'dtype': header[k][1], 'shape': header[k][0]}
             for name, k in tensors.items()}, unmapped)


def render(document):
    return json.dumps(document, indent=2, sort_keys=True) + '\n'


class Hub:
    """Checkpoint lookups, cached per repo so a model shared by several graph trees costs one
    round of requests."""

    def __init__(self):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.cache = {}

    def source(self, repo_id, revision):
        """(source fields, {key: (shape, dtype)}) for `repo_id` at `revision` (None = latest),
        or raise LookupError with a reason the checkpoint is unusable."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
        cache_key = (repo_id, revision)
        if cache_key not in self.cache:
            try:
                info = self.api.model_info(repo_id, revision=revision, files_metadata=True)
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                raise LookupError(f'{repo_id}: {type(e).__name__}') from None
            sibling = next((s for s in info.siblings or [] if s.rfilename == FILENAME), None)
            if sibling is None or sibling.lfs is None:
                raise LookupError(f'no {FILENAME} in {repo_id}')
            metadata = self.api.parse_safetensors_file_metadata(repo_id, FILENAME, revision=info.sha)
            self.cache[cache_key] = ({
                'repo_id': repo_id,
                'revision': info.sha,
                'filename': FILENAME,
                'url': hf_hub_url(repo_id, FILENAME, revision=info.sha),
                'sha256': sibling.lfs.sha256,
                'size': sibling.lfs.size,
            }, {key: (list(t.shape), t.dtype) for key, t in metadata.tensors.items()})
        return self.cache[cache_key]


def _read_unmapped(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get('models') or {}


def _write_unmapped(path, unmapped):
    header = (
        '# Models whose manifest names a Hub checkpoint that their committed graph cannot be\n'
        '# mapped onto, and why -- so `make models.weights.verify` can tell a model that has no\n'
        '# models/<name>/models/safetensors.json on purpose from one that lost it.\n'
        '#\n'
        '# Generated by scripts/weights_map.py -- regenerate with `make models.weights`.\n'
        '\n'
    )
    with open(path, 'w') as f:
        f.write(header)
        f.write(yaml.safe_dump({'models': dict(sorted(unmapped.items()))}, sort_keys=False, width=100))


def cmd_build(args):
    hub = Hub()
    unmapped = {}
    written = 0
    for manifest, models_dir, prefix in args.tree:
        for name, entry in sorted(load_manifest(manifest).items()):
            model_dir = os.path.join(models_dir, name)
            path = os.path.join(model_dir, WEIGHTS_MAP)
            repo_id = entry.get('hf_hub_id')
            if not os.path.isdir(model_dir):
                continue
            if not repo_id:
                if os.path.exists(path):
                    os.remove(path)
                continue
            # A pinned revision is kept unless asked to refresh, so a rebuild after a graph
            # change does not also move every checkpoint.
            revision = None
            if os.path.exists(path) and not args.refresh:
                with open(path) as f:
                    previous = json.load(f)
                if previous['source']['repo_id'] == repo_id:
                    revision = previous['source']['revision']
            try:
                source, header = hub.source(repo_id, revision)
                tensors, missing = map_tensors(_graph_tensors(model_dir, 'weights'),
                                               _graph_tensors(model_dir, 'constants'), header,
                                               prefix)
            except (LookupError, ValueError) as e:
                unmapped[name] = str(e)
                if os.path.exists(path):
                    os.remove(path)
                continue
            document = {'schema_version': 1, 'source': source, 'tensors': tensors,
                        'unmapped': missing}
            schema_validate.validate_document(document, 'safetensors', SCHEMAS_DIR)
            with open(path, 'w') as f:
                f.write(render(document))
            written += 1
    _write_unmapped(args.unmapped, unmapped)
    print(f'Wrote {written} safetensors.json; {len(unmapped)} models unmapped '
          f'({os.path.basename(args.unmapped)})', file=sys.stderr)
    return 0


def verify_model(model_dir, repo_id, prefix=''):
    """Problems with one committed safetensors.json against the graph beside it."""
    from huggingface_hub import hf_hub_url
    with open(os.path.join(model_dir, WEIGHTS_MAP), 'rb') as f:
        document = opgraph.strict_json_loads(f.read())
    schema_validate.validate_document(document, 'safetensors', SCHEMAS_DIR)
    source = document['source']
    problems = []
    if source['repo_id'] != repo_id:
        problems.append(f'repo_id {source["repo_id"]} != manifest hf_hub_id {repo_id}')
    if source['url'] != hf_hub_url(source['repo_id'], source['filename'], revision=source['revision']):
        problems.append(f'url {source["url"]} does not match repo_id/filename/revision')
    weights = _graph_tensors(model_dir, 'weights')
    constants = _graph_tensors(model_dir, 'constants')
    header = {t['key']: (t['shape'], t['dtype']) for t in document['tensors'].values()}
    for name, t in document['tensors'].items():
        if prefix + t['key'] != name:
            problems.append(f'{name}: maps to {t["key"]}, not the name less {prefix!r}')
    try:
        tensors, missing = map_tensors(weights, constants, header, prefix)
    except ValueError as e:
        return problems + [str(e)]
    if set(tensors) != set(document['tensors']):
        problems.append(f'maps tensors the graph does not have: '
                        f'{sorted(set(document["tensors"]) - set(tensors))}')
    if missing != document['unmapped']:
        problems.append(f'unmapped {document["unmapped"]} != {missing}')
    return problems


def cmd_verify(args):
    listed = _read_unmapped(args.unmapped)
    problems, checked = [], 0
    for manifest, models_dir, prefix in args.tree:
        for name, entry in sorted(load_manifest(manifest).items()):
            model_dir = os.path.join(models_dir, name)
            if not os.path.isdir(model_dir):
                continue  # no graph to map, e.g. a model whose autocast export fails
            label = os.path.relpath(model_dir, REPO_ROOT)
            has_map = os.path.exists(os.path.join(model_dir, WEIGHTS_MAP))
            if not entry.get('hf_hub_id'):
                if has_map:
                    problems.append(f'{label}: has {WEIGHTS_MAP} but no hf_hub_id in the manifest')
                continue
            if name in listed:
                if has_map:
                    problems.append(f'{label}: has {WEIGHTS_MAP} but is listed as unmapped')
                continue
            if not has_map:
                problems.append(f'{label}: no {WEIGHTS_MAP} and not listed as unmapped')
                continue
            problems += [f'{label}: {p}'
                         for p in verify_model(model_dir, entry['hf_hub_id'], prefix)]
            checked += 1
    if problems:
        print(f'{len(problems)} problem(s):', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        print('\nRun `make models.weights` to regenerate, and review the diff.', file=sys.stderr)
        return 1
    print(f'{checked} safetensors.json match their graphs; {len(listed)} models listed unmapped',
          file=sys.stderr)
    return 0


def cmd_check_values(args):
    """Load each mapped model through timm's own pretrained path and compare every mapped
    tensor bit for bit against the pinned checkpoint -- the check that timm's load-time
    checkpoint filter does not transform what the map says to copy verbatim.

    With HF_HOME set, downloads go to (and are reused from) that cache. Without it each model
    downloads into its own scratch cache, deleted before the next: all of them together are
    several GB."""
    import contextlib
    import tempfile

    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    seen, failures = set(), 0
    for manifest, models_dir, prefix in args.tree:
        for name in sorted(load_manifest(manifest)):
            path = os.path.join(models_dir, name, WEIGHTS_MAP)
            if name in seen or not os.path.exists(path):
                continue
            seen.add(name)
            with open(path) as f:
                document = json.load(f)
            source = document['source']
            shared = os.environ.get('HF_HOME')
            with (contextlib.nullcontext() if shared else
                  tempfile.TemporaryDirectory(prefix='weights_map_')) as cache:
                checkpoint = load_file(hf_hub_download(source['repo_id'], source['filename'],
                                                       revision=source['revision'], cache_dir=cache))
                model = timm.create_model(name, pretrained=True, cache_dir=cache)
            state = {prefix + n: t for n, t in
                     (dict(model.named_parameters()) | dict(model.named_buffers())).items()}
            bad = [n for n, t in document['tensors'].items()
                   if n not in state or not torch.equal(state[n].detach().to(checkpoint[t['key']].dtype),
                                                        checkpoint[t['key']])]
            print(f'{name}: {"ok" if not bad else f"{len(bad)} differ, e.g. {bad[:3]}"}',
                  file=sys.stderr)
            failures += bool(bad)
            del model, checkpoint, state
    print(f'{len(seen) - failures}/{len(seen)} models load exactly the mapped tensors',
          file=sys.stderr)
    return 1 if failures else 0


def main():
    # On each subcommand rather than the top-level parser: a variable-length --tree ahead of
    # the subcommand would swallow its name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--tree', nargs='+', action='append',
                        metavar='MANIFEST MODELS_DIR [PREFIX]',
                        help='a selection manifest, the graph directory built from it, and the '
                             'prefix its tensor names carry over the checkpoint keys (autocast: '
                             '`model.`); repeatable (default: models-selected.yaml models)')
    common.add_argument('--unmapped', default=os.path.join(REPO_ROOT, 'safetensors-unmapped.yaml'))

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build', parents=[common],
                       help='write safetensors.json for every graph (network)')
    p.add_argument('--refresh', action='store_true',
                   help='re-resolve every checkpoint to its latest revision instead of keeping '
                        'the pinned one')
    p.set_defaults(func=cmd_build)

    p = sub.add_parser('verify', parents=[common],
                       help='check every safetensors.json against its graph (offline)')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('check-values', parents=[common],
                       help="compare timm's pretrained load with the mapped tensors (downloads)")
    p.set_defaults(func=cmd_check_values)

    args = parser.parse_args()
    args.tree = args.tree or [[os.path.join(REPO_ROOT, 'models-selected.yaml'),
                               os.path.join(REPO_ROOT, 'models')]]
    if any(len(tree) not in (2, 3) for tree in args.tree):
        parser.error('--tree takes MANIFEST MODELS_DIR [PREFIX]')
    args.tree = [(tree + [''])[:3] for tree in args.tree]
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
