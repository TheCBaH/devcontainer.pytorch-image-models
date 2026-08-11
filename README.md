[![Build](https://github.com/TheCBaH/devcontainer.pytorch-image-models/actions/workflows/build.yml/badge.svg)](https://github.com/TheCBaH/devcontainer.pytorch-image-models/actions/workflows/build.yml)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev+Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/TheCBaH/devcontainer.pytorch-image-models)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/TheCBaH/devcontainer.pytorch-image-models)

# devcontainer.pytorch-image-models

PyTorch model export and inference samples using the PT2 (`.pt2`) format.

## timm model export report

[`models.md`](models.md) reports, for every architecture registered in the
[`timm`](modules/pytorch-image-models) submodule, whether it can be traced with
`torch.export` (random weights, no download from the Hub), its weight size, and its
inference-time FLOPs — grouped by family and sorted from cheapest to most expensive.

## aten operator cross-reference

A models × operations matrix: for every variant that exports, the operators its graph
contains and, for each, the distinct *configurations* it uses — an operator's non-Tensor
schema arguments (stride, padding, groups, eps, dim, …) plus the result dtype/rank, and for
convolutions the kernel size and whether it is depthwise. Cells are `{configuration id: node
count}`, and the operator catalog at the top of each file resolves the ids:

```yaml
models:
  resnet18:
    conv2d.default: {2: 3, 4: 13, 5: 3, 7: 1}
    relu_.default: {1: 17}
```

It is written **once per dialect**, because each describes a genuinely different operator set
rather than one being a subset of another:

| | file | what it is |
|---|---|---|
| ATen | [`ops-aten.yaml`](ops-aten.yaml) / [`ops-aten.md`](ops-aten.md) | the graph `torch.export.export()` hands back — `conv2d`, `linear`, `layer_norm`, `scaled_dot_product_attention` |
| functional ATen | [`ops-func.yaml`](ops-func.yaml) / [`ops-func.md`](ops-func.md) | the same graphs after `run_decompositions(decomp_table={})` — the same composite operators, but no `relu_`, no `add_`, no eval-time `dropout`; this is the graph published under `models/` |
| core ATen | [`ops-core-meta.yaml`](ops-core-meta.yaml) / [`.md`](ops-core-meta.md), [`ops-core-cpu.yaml`](ops-core-cpu.yaml) / [`.md`](ops-core-cpu.md) | the same graphs after `run_decompositions()` — `convolution`, `addmm`, `native_layer_norm`, `bmm` — what a PT2 backend actually lowers |

Across the whole zoo the three vocabularies overlap only partially — 149 distinct operator names
between them, and no one file contains another:

| | operators | vs. ATen | vs. core ATen |
|---|---|---|---|
| ATen | 111 | — | 67 it has that core lacks, 36 core has that it lacks |
| functional ATen | 96 | 23 dropped, 8 added | 46 it has that core lacks, 30 core has that it lacks |
| core ATen | 80 | 36 added, 67 dropped | — |

The functional dialect is the small step (31 names move, and the operator count *falls*, since
`relu_`/`relu` and the view family collapse together); decomposition is the large one. So a backend
author wants the core files, while someone reading a published graph gets the functional one and
does not need to implement mutation. Model selection (below) covers
the union of all three.

### Why there is a functional dialect in between

`run_decompositions()` always re-runs the AOTDispatcher trace, and functionalization is a fixed
part of that trace rather than a decomposition rule. So an *empty* decomposition table buys the
functionalization without any of the lowering: in-place operators become out-of-place ones, an
in-place slice assignment becomes `select_scatter`, eval-time `dropout` disappears instead of
becoming an identity node, and `conv2d`/`linear`/`layer_norm`/`scaled_dot_product_attention` are
still there as themselves.

Two rewrites come along that are not about mutation, and they have the same cause. An empty table
preserves a composite operator only where torch.export can prove it safe to
(`torch._export.utils._check_valid_to_preserve`): the operator's schema must neither mutate nor
alias its arguments, and it must not be tagged `maybe_aliasing_or_mutating`. `reshape`, `flatten`,
`contiguous`, `chunk` and `to` may all return a view, so they are expanded to
`view`/`_unsafe_view`/`split`/`clone`/`_to_copy`; `batch_norm` and `dropout` are tagged, since they
touch running statistics and RNG state, so `batch_norm` expands to
`_native_batch_norm_legit_no_training`. The vocabulary is therefore small, fixed and derivable —
not "whatever the decomposition table happened to contain".

Like the ATen dialect and unlike core ATen, this all happens before backend dispatch, so it gets a
single file rather than one per backend — with one measured exception worth knowing about.
`scaled_dot_product_attention` survives whole, but the *strides* of the tensor it returns are still
the dispatcher's choice, and a `reshape` sitting directly on top of it decomposes by reading them:

```
meta  sdpa out stride (512, 128, 8, 1) → transpose → clone + _unsafe_view
cpu   sdpa out stride (512,   8, 32, 1) → transpose → view
```

So an attention model traced on CPU can differ from the same model traced on meta by a `clone`/
`_unsafe_view` pair per attention block — `sam2_hiera_tiny` differs in exactly this way, 11 blocks
over 22 nodes. The ATen dialect is immune because it never expands `reshape` at all; core ATen has
the same dependence far more pervasively, which is why it *is* split per backend. Here it is narrow
enough to name rather than to model, and `models.md`'s `device` column says which backend traced
each variant.

### Why core ATen is named per backend

`run_decompositions()` runs *after* dispatch, so a composite operator expands into whichever
kernel the dispatcher selected for the device the model was traced on. One
`scaled_dot_product_attention` call decomposes to 20 nodes with a single `permute` on `meta`
and to 22 nodes with three of them on `cpu`; dtype moves it too (f32 → f16 adds `_to_copy`).
The core ATen *vocabulary* is fixed — what varies is the lowering path, so node counts and
layout operators shift and occasionally the arithmetic does.

A file that merged two backends would therefore describe no single lowering at all. Each names
its own, and each model appears in the file for the backend that actually traced it: `meta`
normally, `cpu` for the architectures meta cannot build (the `device` column in `models.md`
says which). Running the sweep on other hardware adds `ops-core-cuda.*` alongside rather than
overwriting anything.

The ATen graph does *not* vary this way, which is why it needs no backend suffix — and why it
is what gets published.

All of it comes from the same export pass as `models.md`, since the export is what costs the
time.

Regenerate everything with:

```bash
make report          # full run, all ~1300 models, ~15 minutes
make report.dry-run  # quick smoke test on a handful of models
```

To regenerate a named subset (comma-separated globs):

```bash
uv run python scripts/export_report.py --filter 'resnet18,efficientnet_b0' --resume
```

## Published PT2 graphs

[`models/`](models) holds the serialized graph of a representative subset of the zoo, taken
straight out of each model's `.pt2` archive:

```
models/<variant>/models/model.json                      # the exported graph
models/<variant>/data/weights/model_weights_config.json # tensor name -> blob, shape, dtype
```

The weight blobs are not committed — they are large and reproducible from timm at any time.
What is committed is the part that describes the architecture: every ATen node, its arguments,
its shapes and dtypes, and which `nn.Module` it came from. `stack_trace` is dropped before
saving, because it is a third of the file and the only part carrying absolute filesystem paths.

These are **functional ATen** graphs — `torch.export.export()` followed by
`run_decompositions(decomp_table={})`. This removes in-place mutation and eval-time dropout
without applying the default core ATen decomposition table. As a result:

- `conv2d`, `linear`, `layer_norm` and `scaled_dot_product_attention` are
  still there as themselves, instead of the two dozen primitives attention expands into;
- in-place operators such as `relu_`, `add_`, and `silu_` become functional operators;
- the graphs remain much smaller than core ATen. `vit_tiny_patch16_224` is 263 nodes / 302 KB
  against 745 nodes / 1055 KB decomposed, and reproduces eager output bit for bit where the
  decomposed graph drifts by ~1e-6.

Functionalization also canonicalizes the view family and has a narrow device dependence around
attention-result strides; `graph-differences.yaml` records any resulting CPU/meta count mismatch.

### Which models, and why

Publishing all ~1300 would be hundreds of megabytes of near-duplicate JSON. Instead
[`models-selected.yaml`](models-selected.yaml) is computed from the reports above plus
[`model-popularity.yaml`](model-popularity.yaml) (HuggingFace Hub downloads per model,
summed across pretrained tags), so the subset is justified rather than chosen by taste:

- **coverage** — a greedy set cover, not over bare operator names but over (operator, call
  configuration) pairs, since most operators carry several recorded configurations (dtype,
  rank, kwargs) and a graph exercising only the commonest one demonstrates less than one that
  also hits its edges. Coverage spans **every** cross-reference — a model gets credit both for
  its ATen, functional ATen, and core ATen units — while cost uses its ATen node count as a
  stable size proxy. Scored by units gained per graph
  node, so the cheapest carrier of a still-uncovered unit wins over a large model that only
  repeats covered ones;
- **family breadth** — then the cheapest so-far-unrepresented architecture family, repeatedly
  until the target count, so the remaining budget stretches over as many families as possible.

Popularity does not decide which coverage need gets a slot next — cost does, on both counts
above. It only decides *which* model fills a slot once several are eligible for it: the tie
inside coverage's greedy step, and the representative chosen for a family with more than one
eligible variant.

Each entry records `phase` (which of the two put it there), `nodes`, `weight_mb`, `downloads`
(when the Hub has seen the model) and whether it ships as a release archive. `include`/
`exclude` at the top of the file are hand-editable and survive regeneration — `include` is
the escape hatch for an (operator, configuration) or architecture the size caps would
otherwise price out, and `uncovered_op_configs` names any that remain.

`--target` (`TARGET=` for the make targets) trades coverage against committed size — bigger
covers more but every model adds ~1.8KB/node of committed JSON regardless of whether it's still
buying much. [`coverage-curve.yaml`](coverage-curve.yaml) is `make models.curve`'s report of
coverage and family breadth at every target in steps of 10, so that trade-off is a number to
look at rather than a guess: **100** is where op-config coverage's gain per 10 models drops
from ~4-6pp to ~2.3pp, so it's the current default. It covers 2067 of the 3074 (operator,
configuration) units the zoo uses across all three dialects, and 40 of 89 families.

```bash
make models.popularity  # refresh model-popularity.yaml from the HuggingFace Hub (needs network)
make models.curve    # report coverage vs. model count in steps of 10, to (re)pick --target from
make models.select   # recompute the subset (seconds; reads the reports, exports nothing)
make models          # export the subset and refresh models/
make models.verify   # cross-check every committed graph against ops-func.yaml
make check-models    # readable diff of models/ vs HEAD
```

### The graphs and `ops-func.yaml` are cross-checked

`make models.verify` compares each committed graph with the operator counts `ops-func.yaml`
recorded for the same model. **92 of 100 match exactly**: the two are
produced by different code, on different runs, on *different devices* — the report sweeps ~1300
architectures and so traces on `meta`, the only way to touch a multi-billion-parameter model
without materializing it, while a `.pt2` must carry real weight blobs and so traces on CPU.

The remaining eight differences are the narrow device dependence documented for functional
ATen: attention-result strides determine whether a following reshape becomes `view` or
`clone` + `_unsafe_view`; two models also omit `_to_copy` on CPU. These are representation
differences rather than mutation or core-ATen lowering.

[`graph-differences.yaml`](graph-differences.yaml) pins those differences, and `models.verify`
fails if a model starts diverging, stops diverging, or diverges differently — so a torch bump
that shifts something shows up as a reviewable diff rather than passing silently. An entry
appearing there unexpectedly therefore requires review.

```bash
make models.differences   # re-record after a torch/timm bump, then review the diff
```

## Releases

Tagging `v*` publishes, per release-tier model, an archive holding

- `<variant>.pt2` — exported with **real pretrained weights**, so it runs;
- `preprocessing.json` — timm's resolved recipe (input size, crop, interpolation, mean/std);
- `expected.json` — the top-5 that model produces for each sample image;
- `inputs.pt` — a `torch.save`d `dict[str, torch.Tensor]`, keyed by image filename, of each
  sample image already preprocessed for this model (post-transform, pre-batch-dim);
- `outputs.pt` — a `torch.save`d `dict[str, torch.Tensor]`, keyed the same way, of the full
  unrounded output for each of those inputs, for exact numerical verification rather than a
  top-5 sanity check.

`outputs.pt` is generated by feeding the model the tensors `torch.load`ed back out of
`inputs.pt`, not the freshly-transformed values still in memory — so verifying a model means
loading `inputs.pt` and `outputs.pt` and comparing, never re-decoding the original JPEGs.

plus **one** shared `images.zip` with the sample images *exactly as they come from the
dataset* — no resize, crop or normalization — and the ImageNet label files. Preprocessing is
per model, but the images are not, so they are published once and combined with whichever
model's `preprocessing.json` you are using — or skip that step entirely and use `inputs.pt`,
which is already in the tensor form this model expects. See `SOURCES.md` inside it for
provenance.

The release tier is the subset of the published models that has fetchable pretrained weights
and fits under the release weight cap.

```bash
make download            # sample images + labels
make models.fetch        # warm the HuggingFace cache (the only step that fetches weights)
make release             # images.zip + one archive per release-tier model
make resnet10t.release   # just one
```

## Development

The repo ships a devcontainer with all dependencies pre-installed. Open it in VS Code Dev Containers or GitHub Codespaces using the badges above.

### Quick start (local)

```bash
make report.dry-run
```
