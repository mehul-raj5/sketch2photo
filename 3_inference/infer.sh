#!/bin/bash
# Inference with the setting of the reported results (test split, seeds 0/1/2).
#
#   bash 3_inference/infer.sh <dataset> <checkpoint_dir> [device]
#     dataset         wildsketch | cufsf | cuhk | fs2k
#     checkpoint_dir  e.g. checkpoints/fs2k/step_0014000 (from the Drive folder, or your own training run)
#     device          default cuda:0
#
# Writes, under outputs/<dataset>/test/:
#   <dataset>_main_seed{0,1,2}/    direct output of the diffusion model
#   xhat_raw/                      frozen-codebook estimate x̂ (only for datasets with fusion weight a > 0)
#   <dataset>_final_seed{0,1,2}/   after the post-hoc stage (fusion / gray) — these are the folders scored in the paper
set -euo pipefail
DS=${1:?dataset}; CKPT=${2:?checkpoint_dir}; DEV=${3:-cuda:0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
export HF_HOME=${HF_HOME:-$ROOT/data/pretrained/hf_cache} TORCH_HOME=${TORCH_HOME:-$ROOT/data/pretrained/torch_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# per dataset: fusion weight a (chosen on val), scoring size, gray output, prompt colour word
case $DS in
  wildsketch) A=0.5;  SIZE=256x256; GRAY="";       COLOUR="colour" ;;
  fs2k)       A=0.25; SIZE=256x256; GRAY="";       COLOUR="colour" ;;
  cuhk)       A=0.0;  SIZE=200x250; GRAY="";       COLOUR="colour" ;;
  cufsf)      A=0.0;  SIZE=200x250; GRAY="--gray"; COLOUR="black and white" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
PROMPT="RAW photo, photorealistic $COLOUR portrait photograph of a real person, head and shoulders, 85mm portrait lens, f/2.8, sharp focus on the eyes, natural skin texture, visible skin pores, high dynamic range, highly detailed, dslr, 8k uhd"
NEG="drawing, sketch, pencil drawing, charcoal, painting, illustration, cartoon, anime, 3d render, CGI, digital art, airbrushed, plastic skin, waxy skin, doll, mannequin, oversaturated, washed out, low contrast, blurry, out of focus, jpeg artifacts, deformed face, asymmetric eyes, watermark, text"
OUT=$ROOT/outputs/$DS/test

# 1. diffusion: SDEdit from x̂ at strength 0.8, Euler ancestral 50 steps, CFG 1.5, ControlNet residual scale 1.0, tone normalisation on, EMA weights
$PY "$ROOT/3_inference/generate.py" --checkpoint "$CKPT" --split test --final --seeds 0 1 2 --run_id ${DS}_main --device "$DEV" --overwrite \
  --set inference.start=xhat --set inference.guidance_scale=1.5 --set inference.sampler=euler_a --set inference.steps=50 \
  --set conditioning.residual_scale=1.0 --set tone.enabled=true --set tone.strength=1.0 --set inference.strength=0.8 \
  --set "prompt.text=$PROMPT" --set "prompt.negative_style=$NEG"

# 2. raw codebook estimate x̂ (needed only when a > 0)
XARG=""
if [ "$A" != "0.0" ]; then
  $PY "$ROOT/3_inference/generate.py" --checkpoint "$CKPT" --xhat_only --split test --final --run_id xhat_raw --device "$DEV" --overwrite \
    --set tone.enabled=false
  XARG="--xhat $OUT/xhat_raw"
fi

# 3. post-hoc stage
for S in 0 1 2; do
  $PY "$ROOT/3_inference/postprocess.py" --src "$OUT/${DS}_main_seed$S" $XARG --out "$OUT/${DS}_final_seed$S" --a $A --size $SIZE $GRAY
done
echo "done: $OUT/${DS}_final_seed{0,1,2}"
