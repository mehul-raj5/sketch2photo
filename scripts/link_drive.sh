#!/bin/bash
# Link the downloaded Google Drive folders (checkpoints/, codebooks/, frozen_models/, tone_stats/ in one directory) into the repository.
#   bash scripts/link_drive.sh <drive_dir>
# Each folder must keep its internal structure (e.g. checkpoints/<dataset>/step_0014000/).
set -euo pipefail
DRIVE=$(cd "${1:?path to the Drive folder}" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/data/codebooks" "$ROOT/data/tone_stats" "$ROOT/checkpoints"
ln -sfn "$DRIVE/frozen_models" "$ROOT/data/pretrained"
for d in "$DRIVE"/codebooks/*_cvq;  do ln -sfn "$d" "$ROOT/data/codebooks/$(basename "$d")"; done
for d in "$DRIVE"/tone_stats/*;      do ln -sfn "$d" "$ROOT/data/tone_stats/$(basename "$d")"; done
for d in "$DRIVE"/checkpoints/*;     do ln -sfn "$d" "$ROOT/checkpoints/$(basename "$d")"; done
# piq's CLIP-IQA reads the CLIP RN50 weights from ~/.cache/clip
mkdir -p "$HOME/.cache"
[ -e "$HOME/.cache/clip" ] || ln -s "$DRIVE/frozen_models/clip" "$HOME/.cache/clip"
echo "linked: data/pretrained, data/codebooks/*, data/tone_stats/*, checkpoints/*"
