# Evaluation assets and input contracts

Discover `manifest-v2.json` first. Use the legacy `manifest.json` path only
when v2 is absent (HTTP 404). Corrupt or unsupported v2 is an error.
Every asset has a size and file SHA256; every graph and optional weight map
has a producer-commit-addressed URL and digest. Model names alone do not
establish that artifacts belong together.

For each model, fetch its indexed `evaluation` and `inputs` assets. The
evaluation ZIP has exactly `contract-v2.json`, `cases.json`,
`preprocessing.json`, `outputs.pt`, and `expected.json`; it contains no
exported program, weights, or inputs. Obtain graph JSON and pinned Hub
weights independently. An absent weight map is explicit `null`; such an
entry does not promise the independent Hub execution path.

Read `graph.user_inputs` and `graph.user_outputs` in signature order.
`inputs.bindings` maps sample names to user input names and describes
`identity` or `unsqueeze(axis=0)`. Captured weights are excluded. Each case
maps every sample name to its tensor key and canonical descriptor.
`outputs.bindings` describes how graph results become references: the
classification population selects axis 0, index 0, then casts to float32.
Reject incomplete, duplicate or unsupported bindings. Never evaluate a
documentation call signature as code.

Compare full output tensors using
`abs(actual-reference) <= atol + rtol*abs(reference)` from the contract.
All input and output floats must be finite. Rounded top-five JSON is for
display, with ties ordered by class index, and is not the numeric reference.

## Decoded images and preprocessing

The shared `images.pt` is a dictionary of contiguous CPU RGB uint8
`[3,H,W]` tensors at each image's original dimensions. Decode reads bytes
from the verified `images.zip`, calls Pillow `convert('RGB')`, and does
not apply EXIF orientation. `images-metadata.json` preserves source member
and JPEG hashes, decoder/Pillow provenance, source attribution, labels,
sample order, and tensor descriptors. The current ten samples contain
8,466,540 raw decoded bytes.

Packing reloads `images.pt` and reconstructs RGB PIL images before
executing timm's pinned transform. A uint8 tensor passed directly to
`MaybeToTensor` would skip scaling and is not equivalent. The original
JPEG directory is not a packing dependency.

`preprocessing.json` keeps the old fields and adds a versioned `recipe`.
It records the resolved config, actual transform classes, interpolation,
scale size, rounding and padding rules, library versions, timm revision,
helper digest and platform. Each case records the resize, padding and
crop calls observed during the actual transform. Resize operates on uint8
pixels. Conversion/normalization performs distinct float32 division by
255, subtraction of float32 mean, and division by float32 standard deviation.
An independent implementation must also match Pillow's integer coefficient
and rounding arithmetic; geometric agreement alone is insufficient.

Tensor SHA256 covers contiguous row-major logical element bytes in
little-endian order. Dtype and shape are separately mandatory. Only uint8
and float32 are supported; signed zero is bit-significant. File hashes
also cover serialization and are distinct from tensor hashes.

## Producer verification and migration

`make release.manifest` builds the full release tier, creates both indices
and existing reports, independently reproduces staged inputs, replays
freshly loaded exported programs, and verifies all asset relationships
before writing checksums. Run `make release.inputs.verify` separately to
reproduce all authoritative inputs from published decoded images and
recipes. It uses evaluation archives and standalone inputs and can run
with the full model archives removed. A mismatch reports model, case and
first differing element/byte.

`make release.assets` checks the exact indexed file set, hashes, sizes,
ZIP member lists and v1/v2 agreement. Publication invokes it twice, with
the second pass pinned to the first pass's checksum-file digest. The
manual build workflow's `release-validation` mode runs the same DAG and
both gates without creating a tag or release.

The full archive retains `contract.json` version 1 and adds v2 metadata.
Common full/evaluation members have identical bytes; the standalone input
file equals the full archive's `inputs.pt`. Legacy `images_asset_sha256`
still hashes `images.zip`. V2's `decoded_images.sha256` hashes `images.pt`.
The tiny pretrained integration test exercises the existing v1 reader and
playback alongside the v2 path.

Consumers should cache by content digest, verify size/hash before atomic
promotion, and check the graph identity before interpreting bindings.
The first consumer path uses authoritative inputs. Fetching only shared
decoded images and evaluation archives requires an independently verified,
byte-exact preprocessing implementation. Standalone inputs remain the
authority for diagnosing mismatches. Download accounting must include
graph/map metadata and pinned Hub weights, including reuse across models;
the old 2.374 GB archive total is not the amount eliminated by v2.
