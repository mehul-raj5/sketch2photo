#!/bin/bash
# Score the three seeds of one dataset and print the seed mean (FID, KID, LPIPS-VGG, FSIM, ID, BRISQUE, CLIP-IQA).
#
#   bash 4_evaluation/evaluate.sh <dataset> [run_prefix] [device] [split]
#     run_prefix  default <dataset>_final  ->  outputs/<dataset>/<split>/<run_prefix>_seed{0,1,2}
#     split       test (default) | val
set -euo pipefail
DS=${1:?dataset}; PREFIX=${2:-${1}_final}; DEV=${3:-cuda:0}; SPLIT=${4:-test}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
export TORCH_HOME=${TORCH_HOME:-$ROOT/data/pretrained/torch_cache}
RUNS=()
for S in 0 1 2; do
  RID=${PREFIX}_seed$S; PRED=$ROOT/outputs/$DS/$SPLIT/$RID
  $PY "$ROOT/4_evaluation/evaluate.py" --dataset $DS --split $SPLIT --method ours --run_id $RID --pred_dir "$PRED" --device "$DEV" --overwrite
  $PY "$ROOT/4_evaluation/realism.py" --dataset $DS --split $SPLIT --method ours --run_id $RID --pred_dir "$PRED" --device "$DEV"
  RUNS+=($RID)
done
$PY "$ROOT/4_evaluation/summarize.py" --dataset $DS --split $SPLIT --runs "${RUNS[@]}"
