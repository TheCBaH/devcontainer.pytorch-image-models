#!/usr/bin/env bash
# Fetch the sample images and ImageNet label files used by the release archives.
#
# The images are stored and published exactly as they come from the dataset -- no resize,
# no crop, no normalization, no tensor conversion. Each released model carries its own
# preprocessing.json describing how to turn one of these JPEGs into that model's input, so
# the images themselves need to exist only once rather than being baked into every archive.
#
# Usage: ./download.sh [DATA_DIR]
#   DATA_DIR  where to place images/ and labels/  (default: <repo_root>/data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${1:-$REPO_ROOT/data}"
IMAGE_DIR="$DATA_DIR/images"
LABEL_DIR="$DATA_DIR/labels"

# Subset of COCO train2017, packaged by Ultralytics (~7 MB). Generic photographs rather than
# ImageNet validation images, which are not redistributable.
COCO128_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_N="${COCO128_N:-10}"
SYNSETS_URL="https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
METADATA_URL="https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_metadata.txt"

mkdir -p "$IMAGE_DIR" "$LABEL_DIR"

download() {
  local url="$1" dir="$2" name dest
  name="$(basename "$url")"
  dest="$dir/$name"
  if [[ -f "$dest" && -s "$dest" ]]; then
    printf 'Skipping %s (already present)\n' "$name"
    return 0
  fi
  printf 'Downloading %s ...\n' "$name"
  curl -fsSL -A 'pytorch-models-pt2/1.0' -o "$dest" "$url"
}

# --- ImageNet label files ---
# synsets: one WordNet id per line, in the order the 1000 output logits use.
# metadata: synset -> human-readable name, for every synset (not just the 1000).
download "$SYNSETS_URL" "$LABEL_DIR"
download "$METADATA_URL" "$LABEL_DIR"

# --- Sample images ---
# Stamped with the requested count so changing COCO128_N re-extracts, while a repeat run at
# the same count costs nothing.
STAMP="$IMAGE_DIR/.coco128"
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$COCO128_N" ]]; then
  printf 'Skipping sample images (already extracted %s)\n' "$COCO128_N"
else
  printf 'Downloading COCO128 (~7 MB), extracting first %s images ...\n' "$COCO128_N"
  tmp_zip="$(mktemp --suffix=.zip)"
  trap 'rm -f "$tmp_zip"' EXIT
  curl -fsSL -A 'pytorch-models-pt2/1.0' -o "$tmp_zip" "$COCO128_URL"
  # `unzip -j` drops the archive's directory structure; the JPEG bytes are untouched.
  mapfile -t members < <(unzip -Z1 "$tmp_zip" 'coco128/images/train2017/*.jpg' | sort | head -"$COCO128_N")
  unzip -q -j -o "$tmp_zip" "${members[@]}" -d "$IMAGE_DIR"
  printf '%s\n' "$COCO128_N" > "$STAMP"
  printf '  Extracted %s images to %s\n' "${#members[@]}" "$IMAGE_DIR"
fi

# --- Provenance, published alongside the images ---
cat > "$DATA_DIR/SOURCES.md" <<EOF
# Sources

Everything here is redistributed unmodified from its upstream release.

## images/

First ${COCO128_N} JPEGs, in filename order, of \`coco128/images/train2017/\` from:

  ${COCO128_URL}

COCO128 is a 128-image subset of the COCO train2017 dataset packaged by Ultralytics.
The images are byte-identical to the ones in that archive: this repository resizes,
crops and normalizes nothing. Each released model archive carries a
\`preprocessing.json\` describing the transform to apply before inference.

COCO images are licensed under CC BY 4.0; see https://cocodataset.org/#termsofuse

## labels/

  ${SYNSETS_URL}
  ${METADATA_URL}

\`imagenet_lsvrc_2015_synsets.txt\` lists one WordNet synset per line in the order the
1000 ImageNet classifier outputs use, so line N is the class of logit N-1.
\`imagenet_metadata.txt\` maps a synset id to its human-readable name.

From tensorflow/models, Apache License 2.0.
EOF

printf '\nDone: %s\n' "$DATA_DIR"
