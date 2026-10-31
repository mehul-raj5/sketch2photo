# 2_controlnet — stage 2: dual-path ControlNet + SketchAdapter

| Path | What |
|---|---|
| `train_controlnet.py` | step-based, resumable training (`--config` new run · `--resume <run folder>` continue) |
| `configs/<dataset>.yaml` | **final configs** of the reported results (identical recipe on every dataset; only dataset, codebook hashes, landmark file and output size differ) |
| `models/cvqvae_encoder.py` | frozen DECP photo CVQ-VAE (`data/codebooks/<id>_cvq/netG_a_model.pth`, hash-checked): codebook tokens + decoder estimate x̂ |
| `models/sketch_adapter.py` | 16-channel SketchAdapter on the L2-normalised codebook tokens |
| `models/conditioning.py` | 19-channel conditioning (3-ch soft sketch edges + 16-ch adapter) and the ControlNet (init `lllyasviel/sd-controlnet-scribble`) |
| `models/controlnet_pipeline.py` | SD 1.5 (Realistic Vision 5.1 UNet frozen, `sd-vae-ft-mse`, CLIP text encoder), samplers, SDEdit from x̂, tone normalisation |
| `models/face_identity_loss.py`, `models/structure_losses.py` | IR-SE50 identity loss (landmark-aligned), LPIPS-VGG, latent Scharr |
| `data/` | paired training dataset, sketch-only inference dataset, prompts, image I/O |
| `utils/` | config schema + path resolution, checkpoints, EMA, generation loop, PNG writer, tone statistics |
| `tools/precompute_landmarks.py` | MTCNN 5-point landmarks of the training photos (identity-loss alignment) → `outputs/<id>/landmarks/mtcnn5_train.json` |
| `tools/compute_tone_stats.py` | Lab colour statistics of the training photos → `data/tone_stats/<id>/all.json` |

Recipe (all datasets): all training pairs, photo 512², sketch 256², horizontal flip 0.5; ε-MSE with min-SNR (γ 5) and a mixed timestep
sampler; anchored objective (probability 0.6, t 400–850); auxiliary losses on x̂₀ at t ≤ 300: LPIPS-VGG 0.3, IR-SE50 identity 0.2,
latent Scharr L1 0.05; AdamW, LR 1e-5 (ControlNet) / 5e-5 (adapter and new input columns), warm-up 500, cosine to step 10 000, constant
floor to 14 000; batch 8, bf16, gradient checkpointing, EMA 0.999. The reported checkpoint is step 14 000 (EMA weights).
All paths are relative to the repository root (`paths.data_root: data`, `paths.outputs_root: outputs`).
