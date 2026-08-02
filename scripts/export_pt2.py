#!/usr/bin/env python3
"""Build .pt2 archives for the selected timm models and commit their graphs.

A .pt2 file is a zip: a serialized graph (`models/model.json`), an index of the weight
tensors (`data/weights/model_weights_config.json`), and the raw weight blobs. The graph and
the index are small, text, and describe the architecture exactly as a PT2 backend sees it,
so those are extracted into `models/<name>/` and committed; the blobs are not.

`stack_trace` is dropped from every node before saving. It is a third of the serialized
graph, and it is the only part that embeds absolute filesystem paths -- committing it would
make the output differ between a uv checkout and the devcontainer, breaking the "regenerate
and diff" check that gives these files their meaning. What is kept (`nn_module_stack`,
`from_node`, `torch_fn`) is the portable provenance: which module and which pre-dispatch
operator each core ATen node came from.

Commands:
  build     convert + extract every model in the manifest (what `make models` runs)
  convert   one model -> <name>.pt2
  extract   one .pt2 -> models/<name>/ (JSON only)
  fetch     download the release tier's pretrained weights into a shared HF cache
  pack      one release archive: .pt2 + preprocessing.json + expected.json
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import zipfile

import yaml

from exportlib import MAX_RES, cpu_count, resolved_input_size, run_pool, run_worker

# The parts of a .pt2 archive that are worth committing: the graph itself, and the index
# mapping graph tensor names to weight blobs and their shapes/dtypes. Everything else is
# either the blobs (large, and reproducible from timm), a pickled sample input, or archive
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
            'graphs would not be reproducible -- update canonicalize_provenance.')


def load_manifest(path):
    """Read models-selected.yaml into {name: entry}, failing loudly if it is missing."""
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


# ---------------------------------------------------------------------------- worker


def worker_convert(name, output, pretrained, max_res):
    """Export one model and write its .pt2. Runs in its own process (see `run_worker`)."""
    import torch
    import timm

    result = {'name': name, 'status': None, 'error': None}
    try:
        # Real tensors, not `meta`: torch.export.save has to write the weight blobs. The
        # selection caps weight size precisely so this stays affordable.
        model = timm.create_model(name, pretrained=pretrained).eval()
        input_size = resolved_input_size(model.default_cfg, max_res)
        example = torch.randn(1, *input_size)

        # run_decompositions() lowers pre-dispatch ATen (conv2d, batch_norm, linear) to core
        # ATen (convolution, _native_batch_norm_legit_no_training, addmm) -- what a backend
        # actually implements, and what ops.yaml already catalogues for these same models.
        exported = torch.export.export(model, (example,)).run_decompositions()
        make_portable(exported)

        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        torch.export.save(exported, output)

        result['status'] = 'ok'
        result['input_size'] = list(input_size)
        result['nodes'] = sum(1 for n in exported.graph.nodes if n.op == 'call_function')
        result['bytes'] = os.path.getsize(output)
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
    print(json.dumps(result))


def worker_pack(name, model_path, images_dir, output, max_res):
    """Build one release archive: the .pt2, how to preprocess for it, and what it predicts."""
    import torch
    import timm
    from PIL import Image
    from timm.data import create_transform, resolve_data_config
    from timm.models import get_pretrained_cfg

    result = {'name': name, 'status': None, 'error': None}
    try:
        # The registry entry, not a built model: every field needed here is config, and
        # instantiating the architecture would allocate a second full set of weights
        # alongside the ones already loaded from the .pt2.
        pretrained_cfg = get_pretrained_cfg(name)
        cfg = pretrained_cfg.to_dict()

        # The recipe has to describe the resolution the archive was *exported* at, not the
        # model's declared default. Where --max-res caps a 256px or 384px architecture, an
        # export specializes to the capped size and its guards reject anything else, so a
        # sidecar quoting the native size would hand the consumer an input the .pt2 refuses.
        # resolve_data_config indexes pretrained_cfg like a mapping, so it wants the dict
        # form, not the PretrainedCfg object.
        input_size = resolved_input_size(cfg, max_res)
        data_config = resolve_data_config({'input_size': tuple(input_size)}, pretrained_cfg=cfg)

        preprocessing = {
            'model': name,
            'pretrained_tag': cfg.get('tag') or None,
            'input_size': list(data_config['input_size']),
            'crop_pct': data_config.get('crop_pct'),
            'crop_mode': data_config.get('crop_mode'),
            'interpolation': data_config.get('interpolation'),
            'mean': list(data_config['mean']),
            'std': list(data_config['std']),
            'num_classes': cfg.get('num_classes'),
            'timm_version': timm.__version__,
            'torch_version': torch.__version__,
        }

        module = torch.export.load(model_path).module()
        transform = create_transform(**data_config, is_training=False)

        # The images ship once, unmodified, in their own archive; this records what each
        # model makes of them so a consumer can prove they reassembled the pieces correctly.
        expected = {}
        images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for image_name in images:
            tensor = transform(Image.open(os.path.join(images_dir, image_name)).convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                logits = module(tensor)
            top = torch.topk(logits[0].float(), 5)
            expected[image_name] = {
                'top5': [int(i) for i in top.indices],
                'logits': [round(float(v), 4) for v in top.values],
            }

        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, f'{name}.pt2')
            z.writestr('preprocessing.json', json.dumps(preprocessing, indent=2) + '\n')
            z.writestr('expected.json', json.dumps(expected, indent=2) + '\n')

        result['status'] = 'ok'
        result['images'] = len(expected)
        result['bytes'] = os.path.getsize(output)
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
    print(json.dumps(result))


# ---------------------------------------------------------------------------- extract


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
    """{aten target: node count} from a committed graph, for cross-checking against ops.yaml."""
    with open(model_json_path) as f:
        document = json.load(f)
    counts = {}
    for node in document['graph_module']['graph']['nodes']:
        target = node['target'].replace('torch.ops.', '')
        counts[target] = counts.get(target, 0) + 1
    return counts


# ---------------------------------------------------------------------------- commands


def cmd_convert(args):
    worker_convert(args.name, args.output, args.pretrained, args.max_res)


def cmd_extract(args):
    written = extract(args.pt2, args.name, args.models_dir)
    print(f'{args.name}: {len(written)} files -> {os.path.join(args.models_dir, args.name)}',
          file=sys.stderr)


def cmd_build(args):
    """Convert and extract every model in the manifest, one subprocess each."""
    models = load_manifest(args.manifest)
    names = sorted(models)
    if args.limit:
        names = names[:args.limit]

    os.makedirs(args.models_dir, exist_ok=True)

    # A model dropped from the manifest leaves a directory behind that nothing regenerates,
    # so `check-tree-clean` would keep passing on a stale graph forever. Only safe over a
    # full run: a --limit run has no opinion about the models it did not touch.
    if not args.limit:
        for stale in sorted(set(os.listdir(args.models_dir)) - set(names)):
            path = os.path.join(args.models_dir, stale)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f'removed {stale} (no longer in the manifest)', file=sys.stderr)

    # ~3GB of random weight blobs across the tier, written only to be unzipped for the few
    # JSON members and deleted. Unless they are being kept for inspection they never need to
    # touch the repo, so they go somewhere the OS will clean up after a crash.
    keep = args.keep_pt2
    if keep:
        os.makedirs(args.build_dir, exist_ok=True)

    def one(name, scratch):
        pt2 = os.path.join(scratch, f'{name}.pt2')
        # Options declared on the top-level parser have to precede the subcommand.
        result = run_worker(
            __file__,
            ['--max-res', args.max_res, 'convert', name, '--output', pt2],
            name, args.timeout, hf_home=args.hf_home,
        )
        if result.get('status') == 'ok':
            try:
                extract(pt2, name, args.models_dir)
            except Exception as e:
                result = {'name': name, 'status': 'failed', 'error': f'extract: {e}'}
            finally:
                if not keep:
                    os.remove(pt2)
        return result

    with tempfile.TemporaryDirectory(prefix='pt2_build_') as tmp:
        scratch = os.path.abspath(args.build_dir) if keep else tmp
        failures = run_pool(names, lambda name: one(name, scratch), args.workers,
                            lambda r: f' ({r.get("nodes", "?")} nodes)')

    if failures:
        return 1
    print(f'Wrote {len(names)} models to {args.models_dir}', file=sys.stderr)
    return 0


def graph_differences(models, models_dir, ops_path):
    """({name: 'op=reported/committed ...'}, [problem, ...]) comparing graphs with ops.yaml.

    `_assert_tensor_metadata` is dropped on the committed side because the report already
    drops it as export bookkeeping rather than computation -- DROPPED_OPS is that rule, and
    importing it means a second entry there does not silently become 60 spurious diffs here.
    """
    from export_report import DROPPED_OPS, parse_existing_ops, short_op

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
    """The models whose committed graph disagrees with ops.yaml, as a file to commit."""
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


def cmd_verify(args):
    """Hold every committed graph to what ops.yaml says about the same model.

    Where the two agree -- 50 of 60 -- that is real evidence the published graph is the one
    the reports describe: same architecture, different code, different runs. Where they
    disagree the cause is the export device rather than a defect (see render_differences), so
    the divergences are recorded in a file and this checks the recorded set still holds,
    following the same pattern export-exclusions.yaml already uses for the other place a
    measured deviation has to be pinned down rather than argued about.

    That makes both directions failures: a model that starts diverging after a torch bump, and
    one that quietly stops. Either is worth a human looking at the diff.
    """
    models = load_manifest(args.manifest)
    differences, problems = graph_differences(models, args.models_dir, args.ops)

    if args.write:
        with open(args.differences, 'w') as f:
            f.write(render_differences(differences, len(models)))
        print(f'Wrote {args.differences} ({len(differences)} of {len(models)} models differ)',
              file=sys.stderr)
        return 1 if problems else 0

    recorded = read_differences(args.differences)
    for name in sorted(set(differences) | set(recorded)):
        if name not in recorded:
            problems.append(f'{name}: newly differs from ops.yaml -- {differences[name]}')
        elif name not in differences:
            problems.append(f'{name}: no longer differs from ops.yaml (recorded: {recorded[name]})')
        elif differences[name] != recorded[name]:
            problems.append(f'{name}: differs differently\n'
                            f'      recorded: {recorded[name]}\n'
                            f'      now:      {differences[name]}')

    if problems:
        print(f'{len(problems)} problem(s):', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        print(f'\nRun `make models.differences` to re-record, and review the diff.', file=sys.stderr)
        return 1

    print(f'{len(models) - len(differences)}/{len(models)} committed graphs match '
          f'{os.path.basename(args.ops)} exactly; the other {len(differences)} differ exactly as '
          f'{os.path.basename(args.differences)} records', file=sys.stderr)
    return 0


def cmd_fetch(args):
    """Warm a shared HF cache with the release tier's checkpoints.

    Separated from exporting so the export itself runs offline: downloads happen once, in one
    step whose success is legible in a CI log, rather than being attempted by every worker.
    Building the model (rather than just pulling files) also proves the checkpoint loads.
    """
    # This command *is* the online step, so it asserts that itself rather than relying on
    # every caller to remember to unset the flag every other path depends on being set.
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    import timm

    models = load_manifest(args.manifest)
    names = release_names(models, args.only)
    print(f'Fetching {len(names)} checkpoints into {os.environ["HF_HOME"]}', file=sys.stderr)

    def one(name):
        try:
            # Building, rather than just pulling the files, proves the checkpoint actually
            # loads into the architecture -- a corrupt or renamed weight is worth finding
            # here and not when the release archive is already half built.
            del_me = timm.create_model(name, pretrained=True)
            del del_me
            return {'name': name, 'status': 'ok'}
        except Exception as e:
            return {'name': name, 'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:200]}

    # Downloads, so almost entirely network wait: serially this is the longest step in both
    # the release and cache-warming workflows. Distinct repos are safe to fetch at once --
    # huggingface_hub locks per blob and renames atomically.
    return 1 if run_pool(names, one, min(args.workers, len(names) or 1), lambda r: '') else 0


def cmd_pack(args):
    worker_pack(args.name, args.model, args.images, args.output, args.max_res)


def cmd_release(args):
    """Convert with pretrained weights and pack, for every release-tier model."""
    models = load_manifest(args.manifest)
    names = release_names(models, args.only)
    os.makedirs(args.build_dir, exist_ok=True)
    build_dir = os.path.abspath(args.build_dir)

    def one(name):
        pt2 = os.path.join(build_dir, f'{name}.pt2')
        result = run_worker(
            __file__,
            ['--max-res', args.max_res, 'convert', name, '--output', pt2, '--pretrained'],
            name, args.timeout, hf_home=args.hf_home,
        )
        if result.get('status') != 'ok':
            return result
        result = run_worker(
            __file__,
            ['--max-res', args.max_res, 'pack', name, '--model', pt2,
             '--images', os.path.abspath(args.images),
             '--output', os.path.join(build_dir, f'{name}.zip')],
            name, args.timeout, hf_home=args.hf_home,
        )
        os.remove(pt2)
        return result

    failures = run_pool(names, one, args.workers,
                        lambda r: f' ({r["bytes"] / 2**20:.0f} MB)' if r.get('bytes') else '')
    return 1 if failures else 0


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', default=os.path.join(repo_root, 'models-selected.yaml'))
    parser.add_argument('--models-dir', default=os.path.join(repo_root, 'models'))
    parser.add_argument('--build-dir', default=os.path.join(repo_root, '.build'))
    parser.add_argument('--max-res', type=int, default=MAX_RES,
                        help='cap on the traced input resolution, matching the report')
    # `fetch` downloads and the workers read, in separate processes, so they have to name the
    # same directory or the fetch is invisible to the thing it exists to serve. Defaulting to
    # a path inside the repo (rather than the user's real cache) also means CI's actions/cache
    # and a local run behave identically.
    parser.add_argument('--hf-home', default=os.environ.get('HF_HOME') or os.path.join(repo_root, '.hf-cache'),
                        help='shared HuggingFace cache, used by `fetch` and by every worker '
                             '(default: $HF_HOME, else <repo>/.hf-cache)')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build', help='convert + extract every model in the manifest')
    p.add_argument('--workers', type=int, default=cpu_count())
    p.add_argument('--timeout', type=float, default=600.0)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--keep-pt2', action='store_true', help='do not delete the .pt2 after extracting')
    p.set_defaults(func=cmd_build)

    p = sub.add_parser('convert', help='export one model to .pt2')
    p.add_argument('name')
    p.add_argument('--output', required=True)
    p.add_argument('--pretrained', action='store_true')
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser('extract', help='copy the JSON members of a .pt2 into models/<name>/')
    p.add_argument('name')
    p.add_argument('--pt2', required=True)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser('verify', help='cross-check committed graphs against ops.yaml')
    p.add_argument('--ops', default=os.path.join(repo_root, 'ops.yaml'))
    p.add_argument('--differences', default=os.path.join(repo_root, 'graph-differences.yaml'))
    p.add_argument('--write', action='store_true',
                   help='re-record the differences file instead of checking against it')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('fetch', help='download the release tier pretrained weights')
    p.add_argument('--workers', type=int, default=8,
                   help='parallel downloads; network-bound, so not tied to core count')
    p.add_argument('--only', nargs='+', default=None, metavar='MODEL',
                   help='fetch these models instead of the whole release tier')
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser('pack', help='build one release archive')
    p.add_argument('name')
    p.add_argument('--model', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser('release', help='convert + pack every release-tier model')
    p.add_argument('--images', default=os.path.join(repo_root, 'data', 'images'))
    p.add_argument('--workers', type=int, default=cpu_count())
    p.add_argument('--timeout', type=float, default=900.0)
    p.add_argument('--only', nargs='+', default=None, metavar='MODEL',
                   help='build these models instead of the whole release tier')
    p.set_defaults(func=cmd_release)

    args = parser.parse_args()
    # Set before anything imports huggingface_hub, which reads it once at import time.
    os.environ['HF_HOME'] = os.path.abspath(args.hf_home)
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
