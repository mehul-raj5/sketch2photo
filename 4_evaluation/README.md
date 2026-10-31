# 4_evaluation — protocol `eval-v1`

`bash 4_evaluation/evaluate.sh <dataset> [run_prefix] [device]` scores the three seeds and prints their mean.
Results: `4_evaluation/results/<dataset>/test/ours/<run_id>/{summary.json, per_image.csv, realism.json}`; one line per run in `RUNS.md`
/ `RUNS_realism.md`.

Predictions and ground truth are squashed (PIL bicubic) to the scoring size: 256 × 256 for WildSketch and FS2K, 200 × 250 for CUFS and CUFSF.

| Metric | Implementation |
|---|---|
| FID | `pytorch-fid` Inception pool3 (2 048-d), batch 32 — as DECP's evaluation script |
| KID | polynomial kernel, unbiased MMD², 100 subsets of up to 1 000 |
| LPIPS | LPIPS v0.1, VGG (headline); AlexNet and SqueezeNet also stored |
| FSIM | Python port of the original `FeatureSIM.m` on gray images (MATLAB `rgb2gray` weights); the original MATLAB code is in `reference/` |
| ID | DECP's evaluation: MTCNN alignment (`decp/evaluate/`), CurricularFace IR-101 embeddings, cosine similarity to the GT photo at its native size |
| BRISQUE, CLIP-IQA | `piq` 0.8.0 (no-reference; CLIP RN50 weights read from `~/.cache/clip`) |

`compare.py` runs paired Wilcoxon signed-rank tests between two methods' per-image scores.
`decp/evaluate/` is DECP's official evaluation package, used read-only (its LPIPS copy has its own `LICENSE`).
