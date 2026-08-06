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

It is written **once per dialect**, because the two describe genuinely different operator
sets rather than one being a subset of the other:

| | file | what it is |
|---|---|---|
| ATen | [`ops-aten.yaml`](ops-aten.yaml) / [`ops-aten.md`](ops-aten.md) | the graph `torch.export.export()` hands back — `conv2d`, `linear`, `layer_norm`, `scaled_dot_product_attention` — and the graph published under `models/` |
| core ATen | [`ops-core-meta.yaml`](ops-core-meta.yaml) / [`.md`](ops-core-meta.md), [`ops-core-cpu.yaml`](ops-core-cpu.yaml) / [`.md`](ops-core-cpu.md) | the same graphs after `run_decompositions()` — `convolution`, `addmm`, `native_layer_norm`, `bmm` — what a PT2 backend actually lowers |

Measured across a 24-model sample, ATen contributes 33 operator names core ATen never shows
and core contributes 24 the ATen graph never shows; only 25 are common to both. So a backend
author wants the core files, someone reading a published graph wants the ATen one, and model
selection (below) covers the union.

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

These are **ATen** graphs — `torch.export.export()` with no `run_decompositions()` — for three
reasons, all measured:

- **they say more.** `conv2d`, `linear`, `layer_norm` and `scaled_dot_product_attention` are
  still there as themselves, instead of the two dozen primitives attention expands into;
- **they do not depend on the machine.** The decomposed graph does (see above), so a committed
  core ATen graph would be one lowering of several, tied to whatever traced it. The ATen graph
  exported on `meta` and on `cpu` is the same graph, which is also what lets `make models.verify`
  compare a CPU-traced archive against a meta-traced report at all;
- **they are smaller and numerically exact.** `vit_tiny_patch16_224` is 263 nodes / 302 KB
  against 745 nodes / 1055 KB decomposed, and reproduces eager output bit for bit where the
  decomposed graph drifts by ~1e-6.

The trade-off, stated plainly: this IR is not functionalized. Published graphs contain in-place
operators (`relu_`, `add_`, `silu_`) and eval-time `dropout` nodes, so a consumer has to handle
mutation. `run_decompositions({})` would functionalize without decomposing, but it reintroduces
the backend dependence and drops `dropout` while adding `_unsafe_view`/`clone`, so it buys less
than it costs.

### Which models, and why

Publishing all ~1300 would be hundreds of megabytes of near-duplicate JSON. Instead
[`models-selected.yaml`](models-selected.yaml) is computed from the reports above plus
[`model-popularity.yaml`](model-popularity.yaml) (HuggingFace Hub downloads per model,
summed across pretrained tags), so the subset is justified rather than chosen by taste:

- **coverage** — a greedy set cover, not over bare operator names but over (operator, call
  configuration) pairs, since most operators carry several recorded configurations (dtype,
  rank, kwargs) and a graph exercising only the commonest one demonstrates less than one that
  also hits its edges. Coverage spans **every** cross-reference — a model gets credit both for
  the ATen units it publishes and for the core ATen units it decomposes to — while cost is
  always its ATen node count, the graph actually committed. Scored by units gained per graph
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
from ~4-6pp to ~2.3pp, so it's the current default. It covers 2059 of the 3062 (operator,
configuration) units the zoo uses across both dialects, and 41 of 89 families.

```bash
make models.popularity  # refresh model-popularity.yaml from the HuggingFace Hub (needs network)
make models.curve    # report coverage vs. model count in steps of 10, to (re)pick --target from
make models.select   # recompute the subset (seconds; reads the reports, exports nothing)
make models          # export the subset and refresh models/
make models.verify   # cross-check every committed graph against ops-aten.yaml
make check-models    # readable diff of models/ vs HEAD
```

### The graphs and `ops-aten.yaml` agree exactly

`make models.verify` compares each committed graph with the operator counts `ops-aten.yaml`
recorded for the same model. **All 100 match**, which is worth more than it sounds: the two are
produced by different code, on different runs, on *different devices* — the report sweeps ~1300
architectures and so traces on `meta`, the only way to touch a multi-billion-parameter model
without materializing it, while a `.pt2` must carry real weight blobs and so traces on CPU.

They agree because both are ATen graphs, and that dialect does not depend on the device. When
the committed graphs were core ATen, 9 of 100 disagreed and had to be pinned as known
divergences: `scaled_dot_product_attention` lowers to a fused CPU kernel whose decomposition
differs from the math path `meta` takes, so attention architectures landed on different counts
for the operators attention expands into (`test_vit4`: `permute` 65 on meta, 83 on CPU;
`sam2_hiera_tiny` differed in `bmm`, `addmm`, `mul`, `sub` and `cat` as well). Publishing the
undecomposed graph removed the whole category — and it is the same fact that makes the core
ATen cross-references per backend.

[`graph-differences.yaml`](graph-differences.yaml) is consequently empty, and `models.verify`
fails if a model starts diverging, stops diverging, or diverges differently — so a torch bump
that shifts something shows up as a reviewable diff rather than passing silently. An entry
appearing there now means something genuinely unexplained.

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