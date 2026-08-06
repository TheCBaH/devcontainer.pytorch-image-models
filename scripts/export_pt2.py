#!/usr/bin/env python3
"""Build .pt2 archives for the selected timm models and commit their graphs.

A .pt2 file is a zip: a serialized graph (`models/model.json`), an index of the weight
tensors (`data/weights/model_weights_config.json`), and the raw weight blobs. The graph and
the index are small, text, and describe the architecture exactly as a PT2 backend sees it,
so those are extracted into `models/<name>/` and committed; the blobs are not.

The graph is the ATen dialect -- what `torch.export.export()` returns, undecomposed -- so it
carries `conv2d`, `linear`, `layer_norm` and `scaled_dot_product_attention` as themselves, and
does not depend on the machine that produced it. See `worker_convert`.

`stack_trace` is dropped from every node before saving, in every graph -- including those
nested inside higher-order ops, which an undecomposed graph keeps. It is a third of the
serialized graph and the only part that embeds absolute filesystem paths, which would make the
output differ between a uv checkout and the devcontainer and break the "regenerate and diff"
check that gives these files their meaning; `assert_portable` re-reads each archive to be sure.
What is kept (`nn_module_stack`, `from_node`, `torch_fn`) is the portable provenance: which
module each ATen node came from.

Commands:
  build     convert + extract every model in the manifest (what `make models` runs)
  convert   one model -> <name>.pt2
  extract   one .pt2 -> models/<name>/ (JSON only)
  fetch     download the release tier's pretrained weights into a shared HF cache
  pack      one release archive: .pt2 + preprocessing.json + expected.json + inputs.pt + outputs.pt
"""
import argparse
import concurrent.futures
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

from exportlib import MAX_RES, resolved_input_size
from pt2_export_core.archive import (
    ARCHIVE_JSON, assert_portable, extract, graph_differences, graph_op_counts, load_manifest,
    make_portable, read_differences, release_names, render_differences,
)
from pt2_export_core.harness import cpu_count, run_pool, run_worker

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

        # The ATen dialect: the graph torch.export hands back, with conv2d, batch_norm, linear
        # and scaled_dot_product_attention still whole. run_decompositions() here would lower
        # those to core ATen and cost twice over -- it discards the operator detail a reader of
        # the graph wants, and it ties the artifact to this machine, since decomposition runs
        # after dispatch. ops-core-<backend>.yaml catalogues those lowerings, one per backend.
        exported = torch.export.export(model, (example,))
        make_portable(exported)

        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        torch.export.save(exported, output)
        assert_portable(output)

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

        # The images ship once, unmodified, in their own archive; preprocessing is per model,
        # so what this model actually consumes is captured here as inputs.pt, per image and
        # pre-batch-dim. Inference then reads its tensors back out of that file (torch.load,
        # not the freshly-transformed value still sitting in memory), so the archive is proven
        # to contain exactly what was run, not just what was intended.
        images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        inputs = {
            image_name: transform(Image.open(os.path.join(images_dir, image_name)).convert('RGB'))
            for image_name in images
        }
        inputs_buffer = io.BytesIO()
        torch.save(inputs, inputs_buffer)

        # outputs.pt is the full, unrounded tensor per image, for exact numerical
        # verification; expected.json is a rounded top-5 of the same values, for a quick
        # glance without loading a tensor file.
        expected = {}
        outputs = {}
        inputs_buffer.seek(0)
        for image_name, image_input in torch.load(inputs_buffer, weights_only=True).items():
            with torch.no_grad():
                logits = module(image_input.unsqueeze(0))
            image_output = logits[0].float()
            outputs[image_name] = image_output
            top = torch.topk(image_output, 5)
            expected[image_name] = {
                'top5': [int(i) for i in top.indices],
                'logits': [round(float(v), 4) for v in top.values],
            }
        outputs_buffer = io.BytesIO()
        torch.save(outputs, outputs_buffer)

        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, f'{name}.pt2')
            z.writestr('preprocessing.json', json.dumps(preprocessing, indent=2) + '\n')
            z.writestr('expected.json', json.dumps(expected, indent=2) + '\n')
            z.writestr('inputs.pt', inputs_buffer.getvalue())
            z.writestr('outputs.pt', outputs_buffer.getvalue())

        result['status'] = 'ok'
        result['images'] = len(expected)
        result['bytes'] = os.path.getsize(output)
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
    print(json.dumps(result))


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


def cmd_verify(args):
    """Hold every committed graph to what ops-aten.yaml says about the same model.

    Where the two agree that is real evidence the published graph is the one the reports
    describe: same architecture, different code, different runs, different machines. Both are
    ATen graphs, which is what makes the comparison meaningful across the meta/CPU divide --
    the report traces on meta and an archive has to trace on CPU to carry real weights, and the
    undecomposed graph is the one that does not vary between them.

    Any residual divergence is recorded in a file and this checks the recorded set still holds,
    following the same pattern export-exclusions.yaml already uses for the other place a
    measured deviation has to be pinned down rather than argued about. That makes both
    directions failures: a model that starts diverging after a torch bump, and one that quietly
    stops. Either is worth a human looking at the diff.
    """
    models = load_manifest(args.manifest)
    ops_name = os.path.basename(args.ops)
    differences, problems = graph_differences(models, args.models_dir, args.ops)

    if args.write:
        with open(args.differences, 'w') as f:
            f.write(render_differences(differences, len(models), ops_name))
        print(f'Wrote {args.differences} ({len(differences)} of {len(models)} models differ)',
              file=sys.stderr)
        return 1 if problems else 0

    recorded = read_differences(args.differences)
    for name in sorted(set(differences) | set(recorded)):
        if name not in recorded:
            problems.append(f'{name}: newly differs from {ops_name} -- {differences[name]}')
        elif name not in differences:
            problems.append(f'{name}: no longer differs from {ops_name} (recorded: {recorded[name]})')
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

    p = sub.add_parser('verify', help='cross-check committed graphs against ops-aten.yaml')
    p.add_argument('--ops', default=os.path.join(repo_root, 'ops-aten.yaml'))
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
