## Sketch2Photo-CodebookDiffusion

PyTorch code for "**Multi-Modal conditioning for Sketch-to-Photo Synthesis using ControlNet and Stable Diffusion**".

Facial sketch → photo synthesis. The sketch is tokenised with a frozen photo-domain CVQ-VAE codebook (stage 1 of DECP), and a 19-channel
dual-path ControlNet (3-channel soft sketch edges + a 16-channel SketchAdapter over the codebook tokens) steers a frozen
Stable Diffusion 1.5 model (Realistic Vision 5.1 UNet + `sd-vae-ft-mse`). At inference the frozen codebook decoder's estimate x̂ is the
SDEdit starting point. A post-hoc stage adds the x̂ detail layer (fusion weight chosen per dataset on the validation split).

The repository runs in the order of the method, one folder per stage:

| Folder | Stage | Entry point |
|---|---|---|
| `0_data_preparation/` | build the dataset folders and the tuning splits | `prepare_cuhk.py`, `prepare_fs2k.py`, `build_cufsf.py`, `make_splits.py` |
| `1_codebook/` | stage 1: photo-domain CVQ-VAE codebook (DECP's official stage-1 code, see `1_codebook/CHANGES.md`) | `train_codebook.py` |
| `2_controlnet/` | stage 2: dual-path ControlNet + SketchAdapter on the frozen codebook and SD 1.5; final configs in `2_controlnet/configs/` | `train_controlnet.py` |
| `3_inference/` | generation with the reported setting + post-hoc stage | `infer.sh` (`generate.py`, `postprocess.py`) |
| `4_evaluation/` | FID, KID, LPIPS, FSIM, ID, BRISQUE, CLIP-IQA | `evaluate.sh` (`evaluate.py`, `realism.py`, `summarize.py`) |
| `scripts/` | end-to-end reproduction and Drive setup | `reproduce.sh`, `link_drive.sh` |

### Requirements

+ Linux, one NVIDIA GPU (training used one A100 40 GB: about 32 GB peak at batch 8, about 1 s per step; inference needs about 13 GB)
+ Python 3.12, PyTorch 2.14 (CUDA 12.6), diffusers 0.40, transformers 5.17 — exact versions in `requirements.txt`

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Prepare data

No images are distributed with this repository. Download each dataset from its authors and build the folder layout
`data/datasets/<id>/{train,test}/{a,b}/` (a = photo, b = sketch, pairs share a file stem):

| id | Dataset | train / test | How |
|---|---|---|---|
| `wildsketch` | WildSketch (Nie et al., Neurocomputing 2022) | 396 / 400 | official split; photos → `a/`, sketches → `b/` (file stems `1`–`800` as released) |
| `cufsf` | CUFSF sketches (CUHK MMLab) + gray FERET photos (NIST) | 250 / 944 | `python 0_data_preparation/build_cufsf.py --variant gray --feret <colorferet root> --apply` (sketch sources in `data/sources/cufsf_cuhk/`, see `0_data_preparation/README.md`) |
| `cuhk` | CUFS, CUHK student subset | 88 / 100 | `python 0_data_preparation/prepare_cuhk.py --photos <dir> --sketches <dir>` |
| `fs2k` | FS2K (Fan et al., MIR 2022) | 1 058 / 1 046 | `python 0_data_preparation/prepare_fs2k.py --fs2k <FS2K root>` (official annotation split) |

Then write the tuning splits (deterministic; the sha256 of every file is in `0_data_preparation/README.md`):
```
python 0_data_preparation/make_splits.py --dataset wildsketch --n-val 40  --seed 0
python 0_data_preparation/make_splits.py --dataset cufsf      --n-val 25  --seed 0
python 0_data_preparation/make_splits.py --dataset cuhk       --n-val 10  --seed 0
python 0_data_preparation/make_splits.py --dataset fs2k       --n-val 106 --seed 0
```

### Prepare pre-trained models, codebooks and checkpoints

Download the folder from [Google Drive — link to be added](https://drive.google.com/) and link it into the repository:
```
bash scripts/link_drive.sh <path/to/Sketch2Photo-CodebookDiffusion_drive>
```
| Drive folder | Linked to | Contents |
|---|---|---|
| `frozen_models/` | `data/pretrained/` | SD 1.5 Realistic Vision 5.1 (noVAE), `sd-vae-ft-mse`, `sd-controlnet-scribble` (ControlNet initialisation) as a Hugging Face cache; IR-SE50 (identity loss); CurricularFace (ID metric); torchvision / pytorch-fid / piq weights; CLIP RN50 (CLIP-IQA) |
| `codebooks/<id>_cvq/` | `data/codebooks/<id>_cvq/` | stage-1 CVQ-VAE per dataset: `netG_a_model.pth` (photo codebook, used by our method) · `netG_b_model.pth` (sketch codebook, stage-1 by-product) |
| `tone_stats/<id>/all.json` | `data/tone_stats/<id>/` | colour statistics of the training photos (tone normalisation of x̂) |
| `checkpoints/<id>/step_0014000/` | `checkpoints/<id>/step_0014000/` | the ControlNet + SketchAdapter checkpoint behind the reported results (one per dataset; EMA weights are used) |

### Inference

```
bash 3_inference/infer.sh <dataset> checkpoints/<dataset>/step_0014000 cuda:0
```
Setting of the reported results: SDEdit start from x̂, strength 0.8, Euler ancestral 50 steps, CFG 1.5, ControlNet residual scale 1.0,
tone normalisation on, fixed camera prompt (no information from the test photos), seeds 0/1/2; post-hoc fusion weight
a = 0.5 (WildSketch), 0.25 (FS2K), 0 (CUFS); CUFSF outputs are converted to gray (its photos are gray).
Results: `outputs/<dataset>/test/<dataset>_final_seed{0,1,2}/`.

### Train

1. Stage 1, photo-domain codebook (DECP's official schedule: 300 + 300 epochs, batch 1). The WildSketch and CUFSF codebooks are DECP's
   official releases; CUFS and FS2K were trained with:
```
python 1_codebook/train_codebook.py --dataset <dataset> --output_path <dataset>_cvq --device cuda:0 [--resume]
mkdir -p data/codebooks/<dataset>_cvq && cp outputs/codebook/<dataset>/<dataset>_cvq/netG_{a,b}_model.pth data/codebooks/<dataset>_cvq/
```
   A newly trained codebook has new hashes: put its sha256 values into `codebook.sha256_file` / `codebook.sha256_embedding` of the config
   (or set both to `null`).

2. Stage 2, ControlNet + SketchAdapter (14 000 steps, resumable; checkpoints every 500 steps under `outputs/<dataset>/checkpoints/<run_id>/`):
```
python 2_controlnet/tools/precompute_landmarks.py --config 2_controlnet/configs/<dataset>.yaml --device cuda:0   # identity-loss alignment
python 2_controlnet/tools/compute_tone_stats.py   --config 2_controlnet/configs/<dataset>.yaml --split all --device cuda:0
python 2_controlnet/train_controlnet.py --config 2_controlnet/configs/<dataset>.yaml --device cuda:0
# after an interruption:
python 2_controlnet/train_controlnet.py --resume outputs/<dataset>/checkpoints/<run_id> --device cuda:0
```

### Test

```
bash 4_evaluation/evaluate.sh <dataset>        # scores outputs/<dataset>/test/<dataset>_final_seed{0,1,2}, prints the seed mean
```
Everything at once (codebook from the Drive, then train → infer → score): `bash scripts/reproduce.sh <dataset> [--train]`.

The paper reports the mean over seeds 0/1/2 on the test split. Diffusion sampling on a different GPU or library build is not bit-exact; expect small deviations.

### Acknowledgements

Stage 1 (`1_codebook/`) and the ID/LPIPS evaluation package (`4_evaluation/decp/evaluate/`) are the official code of
DECP — *Fine-detailed Facial Sketch-to-Photo Synthesis with Detail-enhanced Codebook Priors* (Zhu et al., IEEE TCSVT 2025);
our edits are limited to paths, device selection, resume and one file-lookup bug (`1_codebook/CHANGES.md`).
We also build on Stable Diffusion, Realistic Vision 5.1, ControlNet (`lllyasviel/sd-controlnet-scribble`), LPIPS, pytorch-fid and piq.
