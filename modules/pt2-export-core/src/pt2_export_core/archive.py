"""Utilities for a .pt2 archive: making a serialized graph reproducible across runs/machines,
extracting its committed JSON members, and cross-checking a committed graph against a report's
operator counts.

A .pt2 file is a zip: a serialized graph (`models/model.json`), an index of the weight
tensors (`data/weights/model_weights_config.json`), and the raw weight blobs. The graph and
the index are small, text, and describe the architecture exactly as a PT2 backend sees it, so
those are the parts worth extracting and committing; the blobs are not.
"""
import json
import os
import sys
import zipfile

import yaml

from .catalog import parse_existing_ops, short_op
from .opgraph import DROPPED_OPS

# The parts of a .pt2 archive that are worth committing: the graph itself, and the index
# mapping graph tensor names to weight blobs and their shapes/dtypes. Everything else is
# either the blobs (large, and reproducible from the zoo), a pickled sample input, or archive
# bookkeeping including a serialization id that changes every run.
ARCHIVE_JSON = (
    'models/model.json',
    'data/weights/model_weights_config.json',
    'data/constants/model_constants_config.json',
)


def make_portable(exported):
    """Strip everything from the graph that describes the machine rather than the model.

    A committed graph has to depend only on the architecture: it is regenerated in two
    different CI jobs and diffed against what is in git, so anything reflecting where or when
    it was built makes that check impossible to pass. Two fields do:

      stack_trace     a third of the serialized graph, every frame naming an absolute path
                      inside the venv, which differs between a uv checkout and the devcontainer
      from_node       provenance, tagged with id(node.graph) -- a Python object address, so a
                      different value on every single run

    Done before saving rather than after extracting, so the released .pt2 is as portable as
    the committed JSON, and so the committed bytes stay the serializer's own output.
    """
    for node in exported.graph.nodes:
        node.meta.pop('stack_trace', None)
    _canonicalize_provenance(exported)


def _canonicalize_provenance(exported):
    """Renumber the graph ids inside `from_node` so the same model always serializes the same.

    `from_node` records where each core ATen node came from, and tags every entry with the
    graph it came from -- as `id(node.graph)`, a Python object address. That address is
    different on every run, so the serialized graph would never be byte-identical twice and
    the "regenerate and diff" check these files exist for could never pass.

    The addresses are only ever compared for equality (did these two nodes come from the same
    graph?), so replacing them with 0, 1, 2... in order of first appearance keeps everything
    the field is used for and drops the only part that was never meaningful. Older torch
    versions numbered them this way to begin with.
    """
    ids = {}
    sources = 0

    def visit(source):
        nonlocal sources
        info = getattr(source, 'node_info', None)
        if info is not None:
            sources += 1
            info.graph_id = ids.setdefault(info.graph_id, len(ids))
        # to_dict() memoizes; the stale copy would otherwise be what gets serialized.
        source._dict = None
        for parent in getattr(source, 'from_node', ()) or ():
            visit(parent)

    seen_from_node = False
    for node in exported.graph.nodes:
        for source in node.meta.get('from_node') or ():
            seen_from_node = True
            visit(source)

    # Everything above reaches into torch internals that are explicitly not backward
    # compatible (NodeSource is @compatibility(is_backward_compatible=False), and `_dict` is
    # a private memo). If a rename ever makes this a no-op, the symptom is a graph that
    # silently differs on every run and a CI failure showing one changed line of minified
    # JSON. Fail here instead, where the cause is stated.
    if seen_from_node and not sources:
        raise RuntimeError(
            'from_node metadata is present but carries no node_info: torch has changed '
            'NodeSource and graph ids are no longer being canonicalized. The committed '
            'graphs would not be reproducible -- update _canonicalize_provenance.')


def load_manifest(path):
    """Read a selection manifest (e.g. models-selected.yaml) into {name: entry}, failing
    loudly if it is missing."""
    if not os.path.exists(path):
        sys.exit(f'{path}: not found -- run `make models.select` first')
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    models = document.get('models') or {}
    if not models:
        sys.exit(f'{path}: no models listed')
    return models


def release_names(models, only=None):
    """The release tier, optionally narrowed to `only`. One reader of the manifest's schema.

    `only` is checked against the whole manifest rather than silently intersected, so a typo
    or a graph-only model is an error at the point it was named instead of an empty run that
    reports success.
    """
    tier = [name for name, entry in sorted(models.items()) if entry.get('release')]
    if not only:
        return tier
    unknown = [name for name in only if name not in models]
    if unknown:
        sys.exit(f'not in the manifest: {", ".join(unknown)}')
    graph_only = [name for name in only if name not in tier]
    if graph_only:
        sys.exit('not release-tier (no fetchable weights, or over the weight cap): '
                 f'{", ".join(graph_only)}')
    return list(only)


def extract(pt2_path, name, models_dir):
    """Copy the JSON members of a .pt2 into models/<name>/, byte for byte.

    The archive nests everything under a directory named after the file stem; that prefix is
    stripped so the committed layout is stable regardless of where the .pt2 was built. The
    bytes are the serializer's own output -- nothing is re-encoded here, so a diff in these
    files always means the graph changed, never that the formatting did.
    """
    target = os.path.join(models_dir, name)
    written = []
    with zipfile.ZipFile(pt2_path) as z:
        roots = {member.split('/', 1)[0] for member in z.namelist()}
        if len(roots) != 1:
            raise ValueError(f'{pt2_path}: expected a single top-level directory, got {sorted(roots)}')
        root = roots.pop()
        for relative in ARCHIVE_JSON:
            member = f'{root}/{relative}'
            try:
                data = z.read(member)
            except KeyError:
                continue  # constants config is absent for models with no constant tensors
            path = os.path.join(target, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
            written.append(relative)
    if not written:
        raise ValueError(f'{pt2_path}: no JSON members found')
    return written


def graph_op_counts(model_json_path):
    """{aten target: node count} from a committed graph, for cross-checking against a report's
    ops matrix."""
    with open(model_json_path) as f:
        document = json.load(f)
    counts = {}
    for node in document['graph_module']['graph']['nodes']:
        target = node['target'].replace('torch.ops.', '')
        counts[target] = counts.get(target, 0) + 1
    return counts


def graph_differences(models, models_dir, ops_path):
    """({name: 'op=reported/committed ...'}, [problem, ...]) comparing graphs with a report's
    ops.yaml-shaped operator matrix.

    `_assert_tensor_metadata` is dropped on the committed side because the report already
    drops it as export bookkeeping rather than computation -- DROPPED_OPS is that rule, so a
    second entry there does not silently become spurious diffs here.
    """
    ops_by_model, _, _ = parse_existing_ops(ops_path)
    differences, problems = {}, []

    for name in sorted(models):
        model_json = os.path.join(models_dir, name, 'models', 'model.json')
        if not os.path.exists(model_json):
            problems.append(f'{name}: no committed graph')
            continue
        try:
            committed = graph_op_counts(model_json)
        except Exception as e:
            problems.append(f'{name}: unreadable graph ({e})')
            continue
        for op in DROPPED_OPS:
            committed.pop(op, None)

        reported = {}
        for op, _, count in ops_by_model.get(name, []):
            reported[op] = reported.get(op, 0) + count
        if not reported:
            problems.append(f'{name}: not in {os.path.basename(ops_path)}')
            continue

        if committed != reported:
            differences[name] = ' '.join(
                f'{short_op(op)}={reported.get(op, 0)}/{committed.get(op, 0)}'
                for op in sorted(set(committed) | set(reported))
                if committed.get(op, 0) != reported.get(op, 0))

    return differences, problems


def render_differences(differences, total):
    """The models whose committed graph disagrees with the ops report, as a file to commit."""
    lines = [
        '# Where a committed graph disagrees with the operator counts in ops.yaml.',
        '#',
        '# Generated by `make models.differences`, do not edit by hand. `make models.verify`',
        '# holds the tree to this list: a model that starts or stops diverging, or diverges',
        '# differently, is a failure until it is regenerated here and reviewed in the diff.',
        '#',
        '# These are not defects. The two artifacts are exported on different devices, and',
        '# they have to be: ops.yaml sweeps ~1300 architectures and so traces on `meta`, the',
        '# only way to reach a multi-billion-parameter model without materializing it, while',
        '# a .pt2 must carry real weight blobs and so traces on CPU. Attention is where the',
        '# two part company -- scaled_dot_product_attention lowers to a fused CPU kernel whose',
        '# decomposition differs from the math path meta takes -- so architectures using it',
        '# land on different counts for the ops attention expands into.',
        '#',
        f'# Values are `op=ops.yaml/graph`. {len(differences)} of {total} models differ.',
        '',
        'models:',
    ]
    for name, delta in sorted(differences.items()):
        lines.append(f'  {name}: {delta}')
    lines.append('')
    return '\n'.join(lines)


def read_differences(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    return {name: str(delta) for name, delta in (document.get('models') or {}).items()}
