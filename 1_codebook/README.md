# 1_codebook — stage 1: photo-domain CVQ-VAE codebook

DECP's official stage-1 code (Zhu et al., IEEE TCSVT 2025): `train_codebook.py` + `utils/`. It trains two CVQ-VAEs (photo `netG_a`,
sketch `netG_b`; 1 024 codes × 256 dimensions) with the official schedule (300 epochs at LR 2e-4 + 300 epochs of linear decay, batch 1,
seed 123). Our method uses only the photo model `netG_a_model.pth`, frozen: its encoder and codebook tokenise the sketch, its decoder
gives the estimate x̂.

```
python 1_codebook/train_codebook.py --dataset <dataset> --output_path <dataset>_cvq --device cuda:0 [--resume]
```
Reads `data/datasets/<dataset>/train/{a,b}`; writes `outputs/codebook/<dataset>/<dataset>_cvq/` (official every-50-epoch checkpoints,
`state.pt` every epoch for `--resume`, final `netG_a_model.pth` / `netG_b_model.pth`). Copy the two final files to
`data/codebooks/<dataset>_cvq/`.

Codebooks behind the reported results (Drive folder `codebooks/`): WildSketch and CUFSF = DECP's official releases (CUFSF = DECP's
`cuhk_feret`); CUFS (CUHK student, 88 training pairs) and FS2K (1 058 training pairs) were trained with this script.
Edits to the official code: `CHANGES.md`. Licence and copyright: `LICENSE` (DECP publishes no licence; the code is included with attribution for reproduction only).
