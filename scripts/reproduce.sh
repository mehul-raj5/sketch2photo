#!/bin/bash
# End-to-end reproduction of one dataset's reported row.
#   bash scripts/reproduce.sh <dataset> [--train] [device]
# Without --train: released checkpoint (checkpoints/<dataset>/step_0014000) -> inference -> scores.
# With --train:    landmarks + tone statistics -> ControlNet training (14 000 steps) -> inference -> scores.
# Prerequisites: data/datasets/<dataset> and data/splits/<dataset> built (0_data_preparation/), the Drive folder linked
# (scripts/link_drive.sh). To retrain the codebook as well, run 1_codebook/train_codebook.py first (README, "Train").
set -euo pipefail
DS=${1:?dataset}; TRAIN=""; DEV=cuda:0
for a in "${@:2}"; do if [ "$a" = "--train" ]; then TRAIN=1; else DEV=$a; fi; done
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PYTHON:-python}
export HF_HOME=${HF_HOME:-$ROOT/data/pretrained/hf_cache} TORCH_HOME=${TORCH_HOME:-$ROOT/data/pretrained/torch_cache} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
CFG=2_controlnet/configs/$DS.yaml
if [ -n "$TRAIN" ]; then
  RUN=$(grep -m1 '^  run_id:' $CFG | awk '{print $2}')
  [ -f outputs/$DS/landmarks/mtcnn5_train.json ] || $PY 2_controlnet/tools/precompute_landmarks.py --config $CFG --device $DEV
  [ -f data/tone_stats/$DS/all.json ] || $PY 2_controlnet/tools/compute_tone_stats.py --config $CFG --split all --device $DEV
  if [ -d outputs/$DS/checkpoints/$RUN ]; then
    $PY 2_controlnet/train_controlnet.py --resume outputs/$DS/checkpoints/$RUN --device $DEV
  else
    $PY 2_controlnet/train_controlnet.py --config $CFG --device $DEV
  fi
  CKPT=outputs/$DS/checkpoints/$RUN/step_0014000
else
  CKPT=checkpoints/$DS/step_0014000
fi
bash 3_inference/infer.sh $DS $CKPT $DEV
bash 4_evaluation/evaluate.sh $DS ${DS}_final $DEV
