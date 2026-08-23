#!/usr/bin/env python3
"""Build .pt2 archives for the selected timm models and commit their graphs.

A .pt2 file is a zip: a serialized graph (`models/model.json`), an index of the weight
tensors (`data/weights/model_weights_config.json`), and the raw weight blobs. The graph and
the index are small, text, and describe the architecture exactly as a PT2 backend sees it,
so those are extracted into `models/<name>/` and committed; the blobs are not.

The graph is the functional ATen dialect: `torch.export.export()` followed by
`run_decompositions(decomp_table={})`. This removes mutation without lowering composite
operators such as `conv2d`, `linear`, `layer_norm`, and `scaled_dot_product_attention` to core
ATen. See `worker_convert`.

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
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

from exportlib import MAX_RES, resolved_input_size
from pt2_export_core import compat, manifest as manifest_mod, opgraph, schema_validate, selection
from pt2_export_core.archive import (
    ARCHIVE_JSON, IMAGES_ARCHIVE_PROFILE, SafeZipError, assert_portable, compare_embedded_graph,
    extract, graph_differences, graph_op_counts, load_manifest, make_portable, read_differences,
    release_names, release_pt2_profile, render_differences, safe_zip_open, selected_pt2_profile,
    single_root,
)
from pt2_export_core.harness import cpu_count, run_pool, run_worker

import models_history

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMAS_DIR = os.path.join(REPO_ROOT, 'schemas')

# ---------------------------------------------------------------------------- op_facts.json staging


def _clean_stale_build_artifacts(models_dir, name):
    """Remove any leftover `<name>.building-*`/`<name>.stale-*` directory from a prior
    interrupted `cmd_build` run. Export is deterministic, so simply clearing these and
    rebuilding from scratch is always safe and always converges -- this is the actual
    recovery mechanism (git is the source of truth between builds, not a bespoke journal),
    not something a crash needs a special code path to repair.
    """
    if not os.path.isdir(models_dir):
        return
    for entry in os.listdir(models_dir):
        if entry.startswith(f'{name}.building-') or entry.startswith(f'{name}.stale-'):
            path = os.path.join(models_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)


def _stage_op_facts(building_dir, convert_result):
    """Build, validate, and write op_facts.json into the scratch directory `extract()` just
    staged model.json into -- entirely before either file ever reaches the committed
    `models/<name>/` directory.

    Bound to the exact staged model.json bytes via model_json_sha256, not just torch_version,
    so a validator can later prove the sidecar still describes the graph actually committed.
    """
    model_json_path = os.path.join(building_dir, 'models', 'model.json')
    with open(model_json_path, 'rb') as f:
        model_json_sha256 = hashlib.sha256(f.read()).hexdigest()

    op_facts = {
        'schema_version': 1,
        'torch_version': convert_result['torch_version'],
        'dialect': 'functional_aten',
        'model_json_sha256': model_json_sha256,
        'ops': convert_result['ops'],
        'schemas': convert_result['schemas'],
    }
    # allow_nan=False as defense in depth: every config value has already gone through
    # opgraph._plain()'s tagged {"$nonfinite_float": ...} shape, so this should never trigger.
    op_facts_bytes = (json.dumps(op_facts, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()

    # Validate in the scratch location, via the same strict parser and schema every later
    # reader uses, before either file ever reaches models/<name>/.
    parsed = opgraph.strict_json_loads(op_facts_bytes)
    if not parsed['ops']:
        raise ValueError('op_facts.json has zero operations -- refusing to stage it')
    schema_validate.validate_document(parsed, 'op_facts', SCHEMAS_DIR)

    with open(os.path.join(building_dir, 'models', 'op_facts.json'), 'wb') as f:
        f.write(op_facts_bytes)


def _swap_into_place(models_dir, name, staged_dir):
    """Publish a staged models/<name>.building-<pid>/ as models/<name>/ -- two renames, not
    one atomic operation. A plain POSIX rename cannot atomically replace a pre-existing
    non-empty directory in one call; a crash between the two renames below leaves
    `models/<name>/` briefly absent on the local working tree. That window is real and is not
    hidden: recovery is git (the last commit is the source of truth between builds, not this
    working tree) plus re-running `make models`, which is always safe since export is
    deterministic.
    """
    live = os.path.join(models_dir, name)
    stale = os.path.join(models_dir, f'{name}.stale-{os.getpid()}')
    if os.path.isdir(live):
        os.replace(live, stale)
    os.replace(staged_dir, live)
    if os.path.isdir(stale):
        shutil.rmtree(stale)


# ---------------------------------------------------------------------------- images.zip validation


def _read_images_archive(images_zip_path):
    """(sha256, {bare image filename: member name}) for `images.zip`, bounded and
    duplicate-checked before anything is trusted from it.

    `images.zip` nests samples under an `images/` prefix and also carries `labels/` and
    `SOURCES.md` (see the Makefile's `zip -r images labels SOURCES.md`); only the
    `images/*.{jpg,jpeg,png}` subset is this contract's concern. The exact-name duplicate
    check runs on the *raw* member list, strictly before any name is normalized into a set --
    a normalized-set comparison would silently collapse a literal duplicate member.
    """
    with open(images_zip_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    with safe_zip_open(images_zip_path, profile=IMAGES_ARCHIVE_PROFILE) as z:
        names = z.names()
        seen = set()
        for name in names:
            if name in seen:
                raise SafeZipError(f'{images_zip_path}: duplicate member name: {name!r}')
            seen.add(name)
        samples = {}
        for name in names:
            if name.startswith('images/') and name.lower().endswith(('.jpg', '.jpeg', '.png')):
                bare = name[len('images/'):]
                if '/' in bare:
                    continue  # non-recursive: a nested subdirectory under images/ is not a sample
                samples[bare] = name
    return sha256, samples


def _local_image_samples(images_dir):
    return sorted(f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def _tensor_layout_fields(tensor):
    return {
        'layout': str(tensor.layout),
        'strides': list(tensor.stride()),
        'storage_offset': tensor.storage_offset(),
        'is_contiguous': tensor.is_contiguous(),
    }


def _tensor_map_contract(tensors, *, member, call_signature, extra_fields=None):
    """contract.json's per-tensor-file record: dtype/shape/keys measured from the actual
    saved tensors, never assumed -- and a `uniform`/`per_key` split for layout/strides/
    storage_offset/is_contiguous, since those can differ per sample even when dtype/shape
    (checked uniform below) do not.
    """
    keys = sorted(tensors)
    dtypes = {str(tensors[k].dtype) for k in keys}
    shapes = {tuple(tensors[k].shape) for k in keys}
    if len(dtypes) != 1 or len(shapes) != 1:
        raise ValueError(f'{member}: dtype/shape are not uniform across samples '
                         f'(dtypes={sorted(dtypes)}, shapes={sorted(shapes)})')

    layouts = {k: _tensor_layout_fields(tensors[k]) for k in keys}
    reference = layouts[keys[0]]
    is_uniform = all(l == reference for l in layouts.values())

    result = {
        'member': member,
        'format': 'torch.save',
        'count': len(keys),
        'keys': keys,
        'dtype': dtypes.pop(),
        'shape': list(shapes.pop()),
        'device': 'cpu',
        'call_signature': call_signature,
        'uniform': is_uniform,
    }
    if is_uniform:
        result.update(reference)
    else:
        result['per_key'] = layouts
    if extra_fields:
        result.update(extra_fields)
    return result


# ---------------------------------------------------------------------------- worker

# Models whose *architecture* (not just its weight values) changes when timm.create_model()
# is called with pretrained=True -- e.g. a factory that does
# `if pretrained: kwargs.setdefault('bn_eps', ...)`. `cmd_build` normally traces every model
# with pretrained=False (cheap, no download) to produce the committed models/<name>/model.json,
# while `cmd_release`'s pack step always uses pretrained=True for the real release-tier .pt2.
# For an ordinary model that only changes which graph is compared to which -- for a model in
# this set the two pipelines would silently diverge (see fbnetc_100, whose exported graph
# bakes in nn.BatchNorm2d's eps as a literal constant, 1e-5 vs 1e-3 depending on the flag),
# breaking verify-release's byte-comparison. Add a name here and re-run `make models` the
# moment such a model is found; test_export_pt2.py pins that cmd_build actually threads
# --pretrained through for every entry, and flags one that has fallen out of the manifest.
PRETRAINED_SENSITIVE_MODELS = {'fbnetc_100'}


def worker_convert(name, output, pretrained, max_res, manifest):
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

        # Functional ATen: the empty table disables optional decompositions while the retrace
        # still functionalizes mutation. Composite operators such as conv2d, linear, and
        # scaled_dot_product_attention remain whole; unlike run_decompositions() with its
        # default table, this does not lower the graph all the way to core ATen.
        exported = torch.export.export(model, (example,))
        exported = exported.run_decompositions(decomp_table={})
        make_portable(exported)

        # Static operator facts for the op_facts.json sidecar, from this same already-built
        # program. A collect_ops() exception here is an ordinary export failure (falls into
        # the except below, status: 'failed') -- never a committed model.json with no
        # matching op_facts.json, since nothing below this line runs if it raises.
        ops, schemas = opgraph.collect_ops(exported)

        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        torch.export.save(exported, output)
        assert_portable(output, selected_pt2_profile(manifest))

        result['status'] = 'ok'
        result['input_size'] = list(input_size)
        result['nodes'] = sum(1 for n in exported.graph.nodes if n.op == 'call_function')
        result['bytes'] = os.path.getsize(output)
        result['ops'] = ops
        result['schemas'] = schemas
        result['torch_version'] = torch.__version__
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
    print(json.dumps(result))


def worker_pack(name, model_path, images_dir, images_archive, images_sha256, output, max_res,
                 manifest):
    """Build one release archive: the .pt2, how to preprocess for it, and what it predicts.

    Both caller-supplied inputs -- `images_archive` and `model_path` -- are validated before
    anything is trusted from them, independently of whatever called this (reachable directly
    via the public `pack` subcommand, not only through `cmd_release`):
      - `images_archive` is bounded/duplicate-checked via `_read_images_archive` (which itself
        goes through `safe_zip_open`), its digest checked against `images_sha256`, and its
        normalized image-sample set compared against `images_dir`'s.
      - `model_path` is bounded/duplicate-checked via `safe_zip_open` before
        `torch.export.load` ever touches it; the exact validated bytes (not a second,
        independent re-read of the path) are what both get deserialized and get embedded in
        the output archive, so the two can never silently diverge.
    """
    import torch
    import timm
    from PIL import Image
    from timm.data import create_transform, resolve_data_config
    from timm.models import get_pretrained_cfg

    result = {'name': name, 'status': None, 'error': None}
    try:
        actual_sha256, archive_samples = _read_images_archive(images_archive)
        if actual_sha256 != images_sha256:
            raise ValueError(f'images.zip sha256 mismatch: expected {images_sha256}, '
                             f'got {actual_sha256}')
        local_samples = _local_image_samples(images_dir)
        if set(archive_samples) != set(local_samples):
            only_archive = sorted(set(archive_samples) - set(local_samples))
            only_local = sorted(set(local_samples) - set(archive_samples))
            raise ValueError(f'images.zip vs {images_dir} sample mismatch -- '
                             f'only in archive: {only_archive}; only in images_dir: {only_local}')

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

        def _read_and_load(handle, proc_fd_path):
            handle.seek(0)
            pt2_bytes = handle.read()
            handle.seek(0)
            try:
                exported = torch.export.load(handle)
            except TypeError:
                # This torch version's export.load rejects a file-like object -- fall back to
                # a race-safe path handoff (the kernel resolves straight back to this same
                # already-open file description) rather than a second open(model_path).
                exported = torch.export.load(proc_fd_path)
            return pt2_bytes, exported

        with safe_zip_open(model_path, profile=release_pt2_profile(manifest)) as zf:
            pt2_bytes, exported = zf._with_validated_source(_read_and_load)
        module = exported.module()
        transform = create_transform(**data_config, is_training=False)

        # The images ship once, unmodified, in their own archive; preprocessing is per model,
        # so what this model actually consumes is captured here as inputs.pt, per image and
        # pre-batch-dim. Inference then reads its tensors back out of that file (torch.load,
        # not the freshly-transformed value still sitting in memory), so the archive is proven
        # to contain exactly what was run, not just what was intended.
        #
        # Each image's own transform+forward runs in its own try/except: a failure carries the
        # lexical image key and how far the loop got, and stops the loop rather than
        # continuing past it -- the fail-closed archive policy is unchanged (no archive is
        # produced on any failure), but the diagnostic now names which image and why.
        inputs = {}
        for completed, image_name in enumerate(local_samples):
            try:
                inputs[image_name] = transform(
                    Image.open(os.path.join(images_dir, image_name)).convert('RGB'))
            except Exception as e:
                raise ValueError(f'image {image_name!r} ({completed}/{len(local_samples)} '
                                 f'preprocessed): {type(e).__name__}: {e}') from e
        inputs_buffer = io.BytesIO()
        torch.save(inputs, inputs_buffer)

        # outputs.pt is the full, unrounded tensor per image, for exact numerical
        # verification; expected.json is a rounded top-5 of the same values, for a quick
        # glance without loading a tensor file. Ranking uses an explicit, deterministic
        # tie-break -- (-logit_value, class_index) -- never torch.topk's unspecified tie
        # order, and any non-finite logit fails packing for that image rather than being
        # silently ranked.
        expected = {}
        outputs = {}
        inputs_buffer.seek(0)
        loaded_inputs = torch.load(inputs_buffer, weights_only=True)
        for completed, (image_name, image_input) in enumerate(loaded_inputs.items()):
            try:
                with torch.no_grad():
                    logits = module(image_input.unsqueeze(0))
                image_output = logits[0].float()
                outputs[image_name] = image_output
                values = [float(v) for v in image_output.tolist()]
                nonfinite = [i for i, v in enumerate(values) if v != v or v in (float('inf'), float('-inf'))]
                if nonfinite:
                    raise ValueError(f'non-finite logit at class index {nonfinite[0]} '
                                     f'for image {image_name!r}')
                ranked = sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
                expected[image_name] = {
                    'top5': ranked,
                    'logits': [round(values[i], 4) for i in ranked],
                }
            except Exception as e:
                raise ValueError(f'image {image_name!r} ({completed}/{len(loaded_inputs)} '
                                 f'run): {e}') from e
        outputs_buffer = io.BytesIO()
        torch.save(outputs, outputs_buffer)

        # The runnable payload contract (suggestion #2): what inputs.pt/outputs.pt/
        # expected.json actually hold, measured from the tensors just saved -- never assumed
        # -- so a consumer never has to guess whether they hold one tensor, a batch, or
        # multiple named samples.
        contract = {
            'schema_version': 1,
            'graph': {'pt2_member': f'{name}.pt2', 'input_size': [1, *data_config['input_size']]},
            'images_asset_sha256': images_sha256,
            'inputs': _tensor_map_contract(inputs, member='inputs.pt', call_signature='model(x)'),
            'outputs': _tensor_map_contract(
                outputs, member='outputs.pt', call_signature='model(x.unsqueeze(0))[0].float()',
                extra_fields={'keys_order': 'identical to inputs',
                             'semantics': 'raw pre-softmax logits'}),
            'expected': {
                'member': 'expected.json',
                'atol': 1e-4,
                'rtol': 1e-3,
                'nonfinite_policy': 'any non-finite position in outputs.pt is an automatic '
                                    'packing failure for that image -- never silently ranked '
                                    'or compared as equal',
                'tie_break': '(-logit_value, class_index)',
            },
            'preprocessing': {
                'member': 'preprocessing.json',
                'provenance': {'timm_version': timm.__version__, 'torch_version': torch.__version__,
                              'pretrained_tag': preprocessing['pretrained_tag']},
            },
            'classes': {'count': cfg.get('num_classes')},
        }
        schema_validate.validate_document(contract, 'contract', SCHEMAS_DIR)

        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            # The exact bytes safe_zip_open validated and torch.export.load deserialized --
            # never a second, independent read of model_path, which could silently embed a
            # different file than the one actually loaded and run above.
            z.writestr(f'{name}.pt2', pt2_bytes)
            z.writestr('preprocessing.json', json.dumps(preprocessing, indent=2, allow_nan=False) + '\n')
            z.writestr('expected.json', json.dumps(expected, indent=2, allow_nan=False) + '\n')
            z.writestr('contract.json', json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n')
            z.writestr('inputs.pt', inputs_buffer.getvalue())
            z.writestr('outputs.pt', outputs_buffer.getvalue())

        result['status'] = 'ok'
        result['images'] = len(expected)
        result['bytes'] = os.path.getsize(output)
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
    print(json.dumps(result))


def _numeric_match(actual, expected, atol, rtol):
    """`actual`/`expected` match under the concrete numeric policy this pipeline uses
    everywhere it compares two tensors: any position where either side is non-finite is an
    automatic mismatch unless both sides are bit-identical non-finite values at that exact
    position (never silently "equal enough"); every other, finite position uses
    `atol=1e-4, rtol=1e-3` (`|a-b| <= atol + rtol*|b|`).
    """
    import torch

    both_finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not torch.equal(actual[~both_finite], expected[~both_finite]):
        return False
    if not both_finite.any():
        return True
    return bool(torch.allclose(actual[both_finite], expected[both_finite], atol=atol, rtol=rtol))


def worker_aoti_attempt(name, pt2_path, release_zip_path, manifest):
    """Best-effort AOTInductor-CPU attempt: `torch.export.load` the just-validated .pt2,
    compile+package it with AOTInductor, load the package, and run it on one real sample from
    the release archive this same pack step just produced -- comparing the *full* output
    tensor (every element, not just top-5) against `outputs.pt` under the same numeric policy
    everywhere else in this pipeline uses.

    Isolated in its own subprocess (see `cmd_release`'s `run_worker` call) so a compiler
    crash/hang/OOM here never costs the pack step's own already-`ok` result -- this worker's
    own failure only ever affects `pack_result['aoti']`, never `pack_result['status']`. Beta
    API, best-effort evidence: a regex-derived "first unsupported op" is recorded as a
    non-authoritative `hint`, never used to change any classification.
    """
    import platform
    import re
    import tempfile

    import torch

    result = {'status': None, 'error': None}
    try:
        with safe_zip_open(pt2_path, profile=release_pt2_profile(manifest)) as zf:
            def _load(handle, proc_fd_path):
                try:
                    return torch.export.load(handle)
                except TypeError:
                    return torch.export.load(proc_fd_path)
            exported = zf._with_validated_source(_load)

        with safe_zip_open(release_zip_path, profile=release_pt2_profile(manifest)) as rz:
            inputs = torch.load(io.BytesIO(rz.read_member('inputs.pt')), weights_only=True)
            outputs = torch.load(io.BytesIO(rz.read_member('outputs.pt')), weights_only=True)
        sample_key = sorted(inputs)[0]
        sample_input = inputs[sample_key].unsqueeze(0)
        expected_output = outputs[sample_key]

        with tempfile.TemporaryDirectory(prefix='aoti_') as cache_dir:
            package_path = torch._inductor.aoti_compile_and_package(
                exported, package_path=os.path.join(cache_dir, f'{name}.pt2'))
            compiled = torch._inductor.aoti_load_package(package_path)
            with torch.no_grad():
                aoti_output = compiled(sample_input)
            if isinstance(aoti_output, (list, tuple)):
                aoti_output = aoti_output[0]
            aoti_output = aoti_output[0].float()

        if not _numeric_match(aoti_output, expected_output, atol=1e-4, rtol=1e-3):
            max_diff = (aoti_output - expected_output).abs().max().item()
            raise ValueError(f'AOTI output diverges from the interpreter output for sample '
                             f'{sample_key!r}: max abs diff {max_diff}')

        result['status'] = 'ok'
        result['environment'] = {
            'torch_version': torch.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
        }
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f'{type(e).__name__}: {e}'[:300]
        hint = re.search(r'aten\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+', str(e))
        if hint:
            result['hint'] = hint.group(0)
    print(json.dumps(result))


# ---------------------------------------------------------------------------- commands


def cmd_convert(args):
    worker_convert(args.name, args.output, args.pretrained, args.max_res, args.manifest)


def cmd_extract(args):
    written = extract(args.pt2, args.name, args.models_dir, selected_pt2_profile(args.manifest))
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
        _clean_stale_build_artifacts(args.models_dir, name)
        pt2 = os.path.join(scratch, f'{name}.pt2')
        # Options declared on the top-level parser have to precede the subcommand. Real
        # weights only for PRETRAINED_SENSITIVE_MODELS -- see its docstring -- so the
        # committed graph matches what cmd_release actually ships for those; everything else
        # stays on the cheap, download-free pretrained=False path.
        convert_args = ['--manifest', args.manifest, '--max-res', args.max_res, 'convert', name,
                         '--output', pt2]
        if name in PRETRAINED_SENSITIVE_MODELS:
            convert_args.append('--pretrained')
        result = run_worker(
            __file__, convert_args, name, args.timeout, hf_home=args.hf_home,
        )
        if result.get('status') == 'ok':
            building = os.path.join(args.models_dir, f'{name}.building-{os.getpid()}')
            try:
                # Both files are staged into a scratch directory and validated there --
                # strict JSON parse, schema, non-empty ops, digest-bound to the staged
                # model.json -- before either ever reaches the committed models/<name>/.
                extract(pt2, os.path.basename(building), args.models_dir,
                        selected_pt2_profile(args.manifest))
                _stage_op_facts(building, result)
                _swap_into_place(args.models_dir, name, building)
            except Exception as e:
                if os.path.isdir(building):
                    shutil.rmtree(building)
                result = {'name': name, 'status': 'failed', 'error': f'stage/swap: {e}'}
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
    """Hold every committed graph to what ops-func.yaml says about the same model.

    Where the two agree that is real evidence the published graph is the one the reports
    describe: same architecture, different code, different runs, different machines. Both are
    functional ATen graphs, which makes the comparison meaningful across the meta/CPU divide.

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
    only = sorted(PRETRAINED_SENSITIVE_MODELS) if args.pretrained_sensitive else args.only
    names = release_names(models, only)
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
    # `pack` is release-tier-only, enforced here rather than by convention: worker_pack's
    # output (a pretrained_cfg-derived preprocessing.json, an expected.json ranking against
    # real predictions) only means anything for a genuine, pretrained, release-tier model, and
    # this is what lets RELEASE_PT2_PROFILE be the always-correct choice for --model below
    # rather than a guess between it and the broader SELECTED_PT2_PROFILE.
    release_names(load_manifest(args.manifest), [args.name])
    worker_pack(args.name, args.model, args.images, args.images_archive, args.images_sha256,
                args.output, args.max_res, args.manifest)


def cmd_aoti_attempt(args):
    worker_aoti_attempt(args.name, args.pt2, args.release_zip, args.manifest)


PACK_RESULTS_FILENAME = 'pack-results.json'


def cmd_release(args):
    """Convert with pretrained weights and pack, for every release-tier model.

    Published output (each model's .zip, images.zip) goes to --release-dir; everything else
    (the scratch .pt2, pack-results.json) goes to --work-dir, which release_assets() never
    scans -- the split that makes "no extra file in the release set" true by construction
    rather than by an allowlist someone could forget to update.

    Writes pack-results.json unconditionally, including on failure -- it is the one place the
    full per-model result (not just a bare error string) survives this process, and is what
    `manifest`/`compat-report` generation reads afterward. A scope: full run (no --only) that
    has any release-tier failure returns non-zero here and is never followed by manifest
    generation in the canonical DAG (see the Makefile) -- no archive-less `unsupported` entry
    is ever published for a full release; --only (smoke scope) is expected to fail sometimes,
    which is exactly what compat-report.json's `unsupported` classification exists for.
    """
    models = load_manifest(args.manifest)
    names = release_names(models, args.only)
    scope = 'smoke' if args.only else 'full'
    release_dir = os.path.abspath(args.release_dir)
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(release_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    images_archive = args.images_archive or os.path.join(release_dir, 'images.zip')
    # Bounded and duplicate-checked once here (IMAGES_ARCHIVE_PROFILE, via
    # _read_images_archive), before its digest is handed to every pack worker; each worker
    # independently repeats the same checks, since `pack` is also reachable directly.
    images_sha256, _ = _read_images_archive(images_archive)

    pack_results = {}

    def one(name):
        pt2 = os.path.join(work_dir, f'{name}.pt2')
        result = run_worker(
            __file__,
            ['--manifest', args.manifest, '--max-res', args.max_res, 'convert', name,
             '--output', pt2, '--pretrained'],
            name, args.timeout, hf_home=args.hf_home,
        )
        if result.get('status') != 'ok':
            pack_results[name] = result
            return result
        release_zip = os.path.join(release_dir, f'{name}.zip')
        result = run_worker(
            __file__,
            ['--manifest', args.manifest, '--max-res', args.max_res, 'pack', name,
             '--model', pt2, '--images', os.path.abspath(args.images),
             '--images-archive', images_archive, '--images-sha256', images_sha256,
             '--output', release_zip],
            name, args.timeout, hf_home=args.hf_home,
        )
        if result.get('status') == 'ok' and not args.skip_aoti:
            # Needs the still-live scratch .pt2 (and the release .zip pack just produced),
            # so this runs strictly before the unconditional os.remove(pt2) below -- an
            # AOTI timeout/crash/OOM only ever affects result['aoti'], never result['status'].
            result['aoti'] = run_worker(
                __file__,
                ['--manifest', args.manifest, 'aoti-attempt', name, '--pt2', pt2,
                 '--release-zip', release_zip],
                name, args.aoti_timeout, hf_home=args.hf_home,
            )
        os.remove(pt2)
        pack_results[name] = result
        return result

    failures = run_pool(names, one, args.workers,
                        lambda r: f' ({r["bytes"] / 2**20:.0f} MB)' if r.get('bytes') else '')

    with open(os.path.join(work_dir, PACK_RESULTS_FILENAME), 'w') as f:
        json.dump({'scope': scope, 'attempted': sorted(names), 'results': pack_results}, f,
                  indent=2, sort_keys=True, allow_nan=False)

    return 1 if failures else 0


def _asset_url(repo, tag, filename):
    """A GitHub release-asset download URL: tag-addressed, digest-verified -- never called
    immutable, since a tag is a movable ref this repo does not control."""
    return f'https://github.com/{repo}/releases/download/{tag}/{filename}'


def cmd_manifest(args):
    """Generate manifest.json, catalogue.json, and compat-report.json into --release-dir from
    a completed `release` run's pack-results.json (in --work-dir) and the committed graphs in
    --models-dir. Requires --repo/--tag/--commit explicitly -- never inferred locally, so the
    same command run in CI or by hand always produces byte-identical URLs for a given input.
    """
    models = load_manifest(args.manifest)
    selected_names = set(models)
    release_tier = set(release_names(models))

    pack_results_path = os.path.join(args.work_dir, PACK_RESULTS_FILENAME)
    if not os.path.exists(pack_results_path):
        sys.exit(f'{pack_results_path}: not found -- run `release` first')
    with open(pack_results_path, 'rb') as f:
        pack_state = opgraph.strict_json_loads(f.read())
    scope = pack_state['scope']
    attempted_names = set(pack_state['attempted'])
    pack_results = pack_state['results']

    if scope == 'full' and release_tier - attempted_names:
        sys.exit(f'{pack_results_path}: scope is "full" but {sorted(release_tier - attempted_names)} '
                 'were never attempted -- refusing to generate a full-scope manifest')
    if scope == 'full' and any(r.get('status') != 'ok' for r in pack_results.values()):
        sys.exit(f'{pack_results_path}: scope is "full" but not every release-tier model '
                 'succeeded -- a full release never publishes a partial manifest')

    roles = selection.load_roles(args.roles)

    model_entries = {}
    for name in sorted(release_tier & attempted_names):
        if pack_results.get(name, {}).get('status') != 'ok':
            continue  # scope: smoke -- an unsupported model simply has no archive/manifest entry
        path = os.path.join(args.release_dir, f'{name}.zip')
        entry = manifest_mod.build_asset_entry(path, _asset_url(args.repo, args.tag, f'{name}.zip'))
        entry['members'] = manifest_mod.archive_members(path, args.manifest)
        if roles.get(name):
            entry['roles'] = roles[name]
        model_entries[name] = entry

    images_path = args.images_archive or os.path.join(args.release_dir, 'images.zip')
    images_entry = manifest_mod.build_asset_entry(images_path, _asset_url(args.repo, args.tag, 'images.zip'))

    history = models_history.load_history(args.history)
    retired = manifest_mod.diff_retirements(release_tier, history)

    timm_version = args.timm_version
    torch_version = args.torch_version
    if timm_version is None or torch_version is None:
        import timm
        import torch
        timm_version = timm_version or timm.__version__
        torch_version = torch_version or torch.__version__
    producer = manifest_mod.build_producer(args.repo, args.tag, args.commit,
                                           timm_version, torch_version)

    manifest_doc = manifest_mod.render_manifest(
        producer=producer, selected_names=selected_names, release_names=release_tier,
        images_entry=images_entry, model_entries=model_entries, retired=retired,
    )
    schema_validate.validate_document(manifest_doc, 'manifest', SCHEMAS_DIR)

    catalogue_doc = manifest_mod.render_catalogue(
        selected_names=selected_names, default_model=args.default_model, repo=args.repo,
        commit=args.commit, roles=roles,
    )
    schema_validate.validate_document(catalogue_doc, 'catalogue', SCHEMAS_DIR)

    compat_doc = compat.render_compat_report(
        scope=scope, selected_names=selected_names, release_names=release_tier,
        attempted_names=attempted_names, models_dir=args.models_dir, pack_results=pack_results,
    )
    schema_validate.validate_document(compat_doc, 'compat-report', SCHEMAS_DIR)

    for filename, doc in (('manifest.json', manifest_doc), ('catalogue.json', catalogue_doc),
                          ('compat-report.json', compat_doc)):
        with open(os.path.join(args.release_dir, filename), 'w') as f:
            json.dump(doc, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    print(f'wrote manifest.json ({len(model_entries)} models), catalogue.json '
         f'({len(selected_names)} models), compat-report.json (scope: {scope}) to '
         f'{args.release_dir}', file=sys.stderr)
    return 0


def cmd_compat_static(args):
    """Static graph facts for every selected model, from op_facts.json alone -- no network, no
    release context, seconds rather than minutes. Fails loudly naming any model whose
    op_facts.json is missing or whose model_json_sha256 no longer matches the committed
    model.json (see compat.static_graph_facts).
    """
    models = load_manifest(args.manifest)
    problems = []
    for name in sorted(models):
        try:
            compat.static_graph_facts(args.models_dir, name)
        except Exception as e:
            problems.append(f'{name}: {e}')
    if problems:
        print(f'{len(problems)} problem(s):', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        return 1
    print(f'{len(models)} models: op_facts.json present and digest-matched', file=sys.stderr)
    return 0


def _validate_release_set_entry(release_zip_path, name, profile, manifest_doc):
    """Cross-document checks a schema alone cannot express: contract.json's
    `images_asset_sha256` against manifest.json's `images.sha256`, its `graph.pt2_member`
    against the archive's own member list, and its declared `inputs`/`outputs` keys/count
    against the tensors actually shipped -- reloaded with `weights_only=True`, and checked to
    actually be a dict of tensors before anything about them is trusted, never assumed from
    the recorded metadata alone.
    """
    import torch

    with safe_zip_open(release_zip_path, profile=profile) as z:
        names = set(z.names())
        contract = opgraph.strict_json_loads(z.read_member('contract.json'))
        schema_validate.validate_document(contract, 'contract', SCHEMAS_DIR)

        if contract['graph']['pt2_member'] not in names:
            raise ValueError(f'contract.json.graph.pt2_member {contract["graph"]["pt2_member"]!r} '
                             f'not present in {release_zip_path}')
        if manifest_doc is not None:
            expected_sha256 = (manifest_doc.get('images') or {}).get('sha256')
            if expected_sha256 and contract['images_asset_sha256'] != expected_sha256:
                raise ValueError('contract.json.images_asset_sha256 does not match '
                                 'manifest.json.images.sha256')

        for side, tensor_member in (('inputs', 'inputs.pt'), ('outputs', 'outputs.pt')):
            data = z.read_member(tensor_member)
            tensors = torch.load(io.BytesIO(data), weights_only=True)
            if not isinstance(tensors, dict) or not all(hasattr(v, 'shape') for v in tensors.values()):
                raise ValueError(f'{tensor_member}: expected a dict of tensors, not '
                                 f'{type(tensors).__name__}')
            declared = contract[side]
            if sorted(tensors) != declared['keys']:
                raise ValueError(f'{tensor_member}: keys {sorted(tensors)} do not match '
                                 f'contract.json.{side}.keys {declared["keys"]}')
            if len(tensors) != declared['count']:
                raise ValueError(f'{tensor_member}: {len(tensors)} tensors, contract.json '
                                 f'declares count={declared["count"]}')


def cmd_verify_release(args):
    """Verify a completed release: every release-tier archive's embedded graph byte-matches
    the committed one, contract.json cross-checks against manifest.json and the shipped
    tensors, and every generated document validates against its schema.
    """
    models = load_manifest(args.manifest)
    release_tier = release_names(models)
    profile = release_pt2_profile(args.manifest)
    problems = []

    manifest_doc = None
    manifest_path = os.path.join(args.release_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'rb') as f:
            manifest_doc = opgraph.strict_json_loads(f.read())

    for name in sorted(release_tier):
        path = os.path.join(args.release_dir, f'{name}.zip')
        if not os.path.exists(path):
            continue  # not part of this (possibly smoke-scope) release
        try:
            compare_embedded_graph(path, name, args.models_dir, profile)
        except Exception as e:
            problems.append(str(e))
        try:
            _validate_release_set_entry(path, name, profile, manifest_doc)
        except Exception as e:
            problems.append(f'{name}: {e}')

    for filename, schema_name in (('manifest.json', 'manifest'), ('catalogue.json', 'catalogue'),
                                  ('compat-report.json', 'compat-report')):
        path = os.path.join(args.release_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as f:
                document = opgraph.strict_json_loads(f.read())
            schema_validate.validate_document(document, schema_name, SCHEMAS_DIR)
        except Exception as e:
            problems.append(f'{filename}: {e}')

    if problems:
        print(f'{len(problems)} problem(s):', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        return 1

    # checksums.txt is generated last, strictly after every check above has passed -- a root
    # checksum list is never produced for artifacts already known to be invalid. It lists
    # every *other* file already in --release-dir; being a checksum file, it does not and
    # cannot list itself.
    relative_paths = []
    for root, _dirs, files in os.walk(args.release_dir):
        for fname in files:
            relative_paths.append(os.path.relpath(os.path.join(root, fname), args.release_dir))
    checksums_text = manifest_mod.render_checksums(args.release_dir, relative_paths)
    with open(os.path.join(args.release_dir, manifest_mod.CHECKSUMS_FILENAME), 'w') as f:
        f.write(checksums_text)

    print(f'verify-release: {len(release_tier)} release-tier archives checked, all OK -- wrote '
         f'{manifest_mod.CHECKSUMS_FILENAME} ({len(relative_paths)} assets)', file=sys.stderr)
    return 0


def cmd_release_assets(args):
    """Re-hash and re-verify --release-dir's asset list against checksums.txt.

    Prints the verified path list (checksums.txt first) to stdout, one per line, followed by
    exactly one final line `checksums_sha256=<digest>` -- everything but the last line is a
    subject path (for `actions/attest-build-provenance`'s `subject-path` or `gh release
    create`'s asset list); the last line is the captured digest a second, pinned invocation
    checks against. Both on stdout, in this fixed shape, so a CI step can split them with
    plain shell (`sed '$d'` / `tail -n1`) without needing a second command or a stderr
    capture. Never trusts an in-memory list from a separate CI process or an earlier
    verification -- see pt2_export_core.manifest.release_assets for the full contract.
    """
    checksums_sha256, paths = manifest_mod.release_assets(
        args.release_dir, expect_checksums_sha256=args.expect_checksums_sha256)
    for path in paths:
        print(path)
    print(f'checksums_sha256={checksums_sha256}')
    return 0


def main():
    repo_root = REPO_ROOT
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

    p = sub.add_parser('verify', help='cross-check committed graphs against ops-func.yaml')
    p.add_argument('--ops', default=os.path.join(repo_root, 'ops-func.yaml'))
    p.add_argument('--differences', default=os.path.join(repo_root, 'graph-differences.yaml'))
    p.add_argument('--write', action='store_true',
                   help='re-record the differences file instead of checking against it')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('fetch', help='download the release tier pretrained weights')
    p.add_argument('--workers', type=int, default=8,
                   help='parallel downloads; network-bound, so not tied to core count')
    fetch_which = p.add_mutually_exclusive_group()
    fetch_which.add_argument('--only', nargs='+', default=None, metavar='MODEL',
                              help='fetch these models instead of the whole release tier')
    fetch_which.add_argument('--pretrained-sensitive', action='store_true',
                              help="fetch just PRETRAINED_SENSITIVE_MODELS -- the real weights "
                                   "cmd_build itself needs -- instead of the whole release tier")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser('pack', help='build one release archive')
    p.add_argument('name')
    p.add_argument('--model', required=True)
    p.add_argument('--images', required=True, help='flat directory of sample images')
    p.add_argument('--images-archive', required=True,
                   help='images.zip -- validated and cross-checked against --images')
    p.add_argument('--images-sha256', required=True,
                   help='expected sha256 of --images-archive')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser('aoti-attempt',
                       help='best-effort AOTInductor-CPU attempt against one release archive')
    p.add_argument('name')
    p.add_argument('--pt2', required=True, help='the (still scratch, not-yet-deleted) .pt2')
    p.add_argument('--release-zip', required=True, help='the release .zip pack just produced')
    p.set_defaults(func=cmd_aoti_attempt)

    p = sub.add_parser('release', help='convert + pack every release-tier model')
    p.add_argument('--images', default=os.path.join(repo_root, 'data', 'images'))
    p.add_argument('--images-archive', default=None,
                   help='images.zip (default: <release-dir>/images.zip)')
    p.add_argument('--release-dir', default=os.path.join(repo_root, '.build', 'release'),
                   help='published output: every model .zip, images.zip -- exactly the '
                        'checksum-listed upload set')
    p.add_argument('--work-dir', default=os.path.join(repo_root, '.build', 'work'),
                   help='scratch .pt2s and pack-results.json -- never scanned by release_assets()')
    p.add_argument('--workers', type=int, default=cpu_count())
    p.add_argument('--timeout', type=float, default=900.0)
    p.add_argument('--aoti-timeout', type=float, default=300.0,
                   help='own budget for the best-effort AOTInductor attempt, independent of '
                        '--timeout')
    p.add_argument('--skip-aoti', action='store_true',
                   help='skip the AOTInductor attempt entirely (still packs normally)')
    p.add_argument('--only', nargs='+', default=None, metavar='MODEL',
                   help='build these models instead of the whole release tier')
    p.set_defaults(func=cmd_release)

    p = sub.add_parser('manifest', help='generate manifest.json/catalogue.json/compat-report.json')
    p.add_argument('--release-dir', default=os.path.join(repo_root, '.build', 'release'))
    p.add_argument('--work-dir', default=os.path.join(repo_root, '.build', 'work'))
    p.add_argument('--images-archive', default=None,
                   help='images.zip (default: <release-dir>/images.zip)')
    p.add_argument('--history', default=os.path.join(repo_root, 'models-history.yaml'))
    p.add_argument('--roles', default=os.path.join(repo_root, 'models-roles.yaml'),
                   help='hand-maintained roles file (suggestion #6); see models-role-candidates.yaml')
    p.add_argument('--repo', required=True, help='e.g. org/repo -- for asset/graph URLs')
    p.add_argument('--tag', required=True, help='the release tag being published')
    p.add_argument('--commit', required=True, help='the full commit SHA being released')
    p.add_argument('--default-model', required=True)
    p.add_argument('--timm-version', default=None)
    p.add_argument('--torch-version', default=None)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser('verify-release',
                       help='byte-compare embedded graphs, validate documents, write checksums.txt')
    p.add_argument('--release-dir', default=os.path.join(repo_root, '.build', 'release'))
    p.set_defaults(func=cmd_verify_release)

    p = sub.add_parser('compat-static',
                       help='static graph facts (op_facts.json only) for every selected model')
    p.set_defaults(func=cmd_compat_static)

    p = sub.add_parser('release-assets',
                       help='re-hash and re-verify --release-dir against checksums.txt')
    p.add_argument('--release-dir', default=os.path.join(repo_root, '.build', 'release'))
    p.add_argument('--expect-checksums-sha256', default=None,
                   help='pin against a prior invocation\'s checksums_sha256 output')
    p.set_defaults(func=cmd_release_assets)

    args = parser.parse_args()
    # Set before anything imports huggingface_hub, which reads it once at import time.
    os.environ['HF_HOME'] = os.path.abspath(args.hf_home)
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
