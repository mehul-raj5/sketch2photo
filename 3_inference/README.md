# 3_inference

`bash 3_inference/infer.sh <dataset> <checkpoint step folder> [device]` runs the three steps of the reported setting:

1. `generate.py` — diffusion from the sketch only: SDEdit start x̂, strength 0.8, Euler ancestral 50 steps, CFG 1.5, residual scale 1.0,
   tone normalisation, fixed camera prompt + negative prompt, EMA weights, seeds 0/1/2 (per-image seeds derived from seed and file stem).
   Only the inference block, prompt texts, `conditioning.residual_scale` / `paths` and `tone` can be overridden (`--set key=value`);
   every override is recorded in each output folder's `run.json`.
2. `generate.py --xhat_only` — the raw codebook estimate x̂ (datasets with fusion weight a > 0).
3. `postprocess.py` — x̂ detail fusion on luminance (σ 1.5, weight a per dataset) and gray output for CUFSF.

`--split test` requires `--final`: tune on `val` only.
