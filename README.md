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

[`ops.yaml`](ops.yaml) is a models × operations matrix: for every variant that exports, the
core ATen operators its graph contains and, for each, the distinct *configurations* it uses
— an operator's non-Tensor schema arguments (stride, padding, groups, eps, dim, …) plus the
result dtype/rank, and for convolutions the kernel size and whether it is depthwise. Cells
are `{configuration id: node count}`, and the operator catalog at the top of the file
resolves the ids:

```yaml
models:
  resnet18:
    convolution.default: {2: 3, 4: 13, 5: 3, 7: 1}
    relu.default: {1: 17}
```

[`ops.md`](ops.md) is the same data op-major and human-first: per operator, which
configurations the zoo needs and how many variants need each — the "what must my backend
implement" view.

Both come from the same export pass as `models.md`, since the export is what costs the
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
What is committed is the part that describes the architecture: every core ATen node, its
arguments, its shapes and dtypes, and which `nn.Module` and pre-dispatch operator it came
from. `stack_trace` is dropped before saving, because it is a third of the file and the only
part carrying absolute filesystem paths.

### Which models, and why

Publishing all ~1300 would be hundreds of megabytes of near-duplicate JSON. Instead
[`models-selected.yaml`](models-selected.yaml) is computed from the reports above, so the
subset is justified rather than chosen by taste:

- **operator coverage** — a greedy set cover over the core ATen operators in `ops.yaml`,
  scored by operators gained per graph node, so the cheapest carrier of a rare operator wins
  over a large model that only repeats common ones;
- **family breadth** — then the smallest architecture family with no representative yet,
  repeatedly, until the target count.

Each entry records `phase` (which of the two put it there), `nodes`, `weight_mb` and whether
it ships as a release archive. `include`/`exclude` at the top of the file are hand-editable
and survive regeneration — `include` is the escape hatch for an operator the size caps would
otherwise price out, and `uncovered_ops` names any that remain.

```bash
make models.select   # recompute the subset (seconds; reads the reports, exports nothing)
make models          # export the subset and refresh models/
make models.verify   # cross-check every committed graph against ops.yaml
make check-models    # readable diff of models/ vs HEAD
```

### The graphs and `ops.yaml` do not always agree

`make models.verify` compares each committed graph with the operator counts `ops.yaml`
recorded for the same model. 50 of the 60 agree exactly. The other 10 differ, and the reason
is the **export device**:

- `ops.yaml` sweeps ~1300 architectures, so it traces on `meta` — the only way to touch a
  multi-billion-parameter model without materializing it;
- a `.pt2` must carry real weight blobs, so it traces on CPU.

`scaled_dot_product_attention` lowers to a fused kernel on CPU whose decomposition differs
from the math path `meta` takes, so architectures using attention land on different counts
for the operators attention expands into. For most that is `permute` alone (`test_vit4`: 65
on meta, 83 on CPU), but where the whole attention block re-decomposes it reaches the
arithmetic too — `sam2_hiera_tiny` differs in `bmm`, `addmm`, `mul`, `sub` and `cat` as well.
Two more (`edgenext_xx_small`, `rdnet_tiny`) differ by a single `as_strided` node, which is a
memory-layout choice rather than attention.

Neither side is wrong: they describe the same architecture exported two ways, and the
committed graph is the one you get if you export it yourself. The full set is pinned in
[`graph-differences.yaml`](graph-differences.yaml), and `models.verify` fails if a model
starts diverging, stops diverging, or diverges differently — so a torch bump that shifts a
decomposition shows up as a reviewable diff rather than passing silently.

```bash
make models.differences   # re-record after a torch/timm bump, then review the diff
```

## Releases

Tagging `v*` publishes, per release-tier model, an archive holding

- `<variant>.pt2` — exported with **real pretrained weights**, so it runs;
- `preprocessing.json` — timm's resolved recipe (input size, crop, interpolation, mean/std);
- `expected.json` — the top-5 that model produces for each sample image.

plus **one** shared `images.zip` with the sample images *exactly as they come from the
dataset* — no resize, crop or normalization — and the ImageNet label files. Preprocessing is
per model, but the images are not, so they are published once and combined with whichever
model's `preprocessing.json` you are using. See `SOURCES.md` inside it for provenance.

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