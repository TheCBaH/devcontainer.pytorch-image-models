# Precision (fp16/bf16 cast and autocast)

Two ways to run a model at lower precision, each a distinct dialect/graph, not two views of
the same one:

| Policy | What changes | Input | Traces on |
| --- | --- | --- | --- |
| `cast` | `model.to(dtype=torch.float16\|bfloat16)` plus a matching input | fp16/bf16 | `meta` (whole ~1300-model catalog; a plain dtype-metadata change a meta tensor represents correctly) |
| `autocast` | `torch.autocast('cpu', dtype=torch.float16\|bfloat16)` forward wrapper around an otherwise-fp32 model and input | fp32 | real CPU tensors only (autocast's dispatch never engages on `meta` -- a meta-traced autocast graph comes back uniformly fp32, indistinguishable from "had no effect") |

This maps directly onto timm's own `validate.py`/`benchmark.py` flags: `cast` is
`--model-dtype float16|bfloat16` (no AMP), `autocast` is `--amp --amp-dtype float16|bfloat16`
(fp32 model/input, AMP on). timm rejects combining the two.

## Autocast is backend-specific, not just a label

Confirmed against the pinned torch 2.13.0+cpu source (`torch/amp/autocast_mode.py`), not
assumed: `torch.autocast` requires `device_type` as its first argument, and CPU/CUDA disagree
on both the default `dtype` (`torch.get_autocast_dtype('cpu')` is `bfloat16`,
`torch.get_autocast_dtype('cuda')` is `float16`) and on which dtypes are even accepted (CUDA
additionally gates `bfloat16` on `torch.cuda.is_bf16_supported()`; CPU only checks membership
in a fixed `[bfloat16, float16]` list) -- on top of each backend registering its own separate
eligible-operator policy, which was not enumerated for either device here. Every autocast
result in this repo -- the `ops-func-autocast-{fp16,bf16}.*` cross-references and the
`models-{fp16,bf16}/autocast/` graphs -- is `torch.autocast('cpu', dtype=...)` specifically;
none of it should be assumed to hold for `torch.autocast('cuda', ...)` without separately
re-running the same checks on a CUDA-enabled machine.

## What's published

| | cast | autocast |
| --- | --- | --- |
| ATen op cross-reference (whole zoo vs. GFLOPs/weight-capped subset) | [`ops-aten-fp16.yaml`](ops-aten-fp16.yaml)/[`.md`](ops-aten-fp16.md), [`ops-aten-bf16.yaml`](ops-aten-bf16.yaml)/[`.md`](ops-aten-bf16.md) | [`ops-func-autocast-fp16.yaml`](ops-func-autocast-fp16.yaml)/[`.md`](ops-func-autocast-fp16.md), [`ops-func-autocast-bf16.yaml`](ops-func-autocast-bf16.yaml)/[`.md`](ops-func-autocast-bf16.md) |
| Representative-subset selection | [`models-fp16.yaml`](models-fp16.yaml), [`models-bf16.yaml`](models-bf16.yaml) | (same file -- one selection covers both policies) |
| Committed graphs | [`models-fp16/cast/`](models-fp16/cast), [`models-bf16/cast/`](models-bf16/cast) | [`models-fp16/autocast/`](models-fp16/autocast), [`models-bf16/autocast/`](models-bf16/autocast) |

`cast`'s cross-reference and selection candidacy cover the whole catalog (meta-only, cheap);
`autocast`'s is restricted to models at or under 50 GFLOPs *and* 150MB fp32 weight (both read
from `models.md`) -- the weight cap matters as much as the GFLOPs one, since a
compute-cheap-but-wide model (e.g. a 35 GFLOPs / 797MB architecture) still saves a large
archive per policy per parallel worker; an uncapped run drove a shared host from ~6GB to ~1GB
free before being capped. See the README's ["Precision (fp16/bf16) graphs"](README.md) section
for the Makefile targets that regenerate any of the above.

## Zoo-wide findings

Full sweep of every `torch.export`-able model (1289 at graph level, 750 at runtime after the
GFLOPs/weight cap), fresh random-weight model per policy (never warm one policy then cast --
a warmed model can leave unregistered positional-index tensors at the wrong dtype).

- **Graph export:** 1238/1289 (96%) cast cleanly at the meta level, identically for fp16 and
  bf16 -- the 51 failures (36 `regnet*`, 7 `efficientvit_mit`, 6 `eva`, 2 `csatv2*`) are
  meta-device/architecture facts, not a property of which low-precision dtype was requested.
- **Runtime:** of the 750 capped models, `cast` succeeds on 735-736/738 and `autocast` on
  733/738, for both dtypes, on the same five models: `vit_small_patch16_rope_mixed_{ape_,}224`
  (`eva`) fails `cast` export on an untouched rotary-embedding buffer; `hiera_small_abswin_256`
  fails `cast` at run (`"compute_index_ranges_weights" not implemented` for that dtype, no CPU
  kernel at all); `mvitv2_{tiny,small,small_cls}` and `sequencer2d_{s,m}` fail `autocast`
  decomposition on a mixed-dtype `linear` call (autocast left one operand fp32, the other low
  precision). These are architectural gaps, not fp16-vs-bf16-specific -- a model that struggles
  with one low-precision dtype struggles with the other.
- **Numeric divergence vs. the fp32 baseline** is the one place the two dtypes genuinely
  differ: fp16's ~65504 finite ceiling produces non-finite output on 46/738 `ok` results (7
  families -- `mobileone`, `dla`, `inception_v3`, `efficientnet`, `visformer`, `hrnet`,
  `mobilenetv3`); bf16 never produces a non-finite output on the same 738 models (its exponent
  range matches fp32's) but reaches ~3 orders of magnitude larger typical-case divergence, and
  its worst cases are large enough to be nonsensical in absolute terms (e.g. `mobileone_s0`:
  max abs error 2.95e13). Both are random-weight artifacts -- an untrained network can
  accumulate instability a pretrained checkpoint would not necessarily reproduce -- not
  evidence either dtype is unusable with real weights; validating that against pretrained
  weights on held-out data is unaddressed here.
- **Autocast's structural cost:** `autocast` graphs mix `{fp16,bf16}`+`f32`+`i64`+`bool`
  (334/738 `ok` autocast graphs keep at least one fp32-output op) where `cast` graphs are
  precision-pure apart from integer/bool metadata; autocast costs 1.38x mean op count (up to
  2.18x) against cast's 1.002x. Archive size moves the opposite way: `cast` averages 51% of the
  fp32 archive (weights actually halve), `autocast` averages 100.3% (same fp32 weights, plus the
  cast nodes' own negligible serialized cost) -- identical between fp16 and bf16, since archive
  size is dictated by stored weight dtype (both 2 bytes/element), not by which 16-bit format.

## Not implemented here

Selective mixed precision (explicit per-module/op FP32 overrides) and quantization (TorchAO
INT8/INT4 weight-only, PT2E static INT8) were investigated as designs/prototypes but are not
part of this repo's build -- see the exporter's own code comments for what `cast`/`autocast`
actually run, and the [PyTorch AMP reference](https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md)
and [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) for the
general background.
