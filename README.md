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

## Development

The repo ships a devcontainer with all dependencies pre-installed. Open it in VS Code Dev Containers or GitHub Codespaces using the badges above.

### Quick start (local)

```bash
make report.dry-run
```