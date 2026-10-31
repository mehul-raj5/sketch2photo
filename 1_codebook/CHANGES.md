# Changes to DECP's official stage-1 code

Only paths, device selection, resume and one file-lookup bug were changed; networks, losses, schedule and seed are the official ones.

| File | Change | Numerical effect |
|---|---|---|
| `train_codebook.py` | **Paths:** data from `data/datasets/<id>/train/{a,b}` (`--data_root`), outputs to `outputs/codebook/<id>/<output_path>/` (`--out_dir`). **Resume:** `state.pt` every epoch (`--resume`), the official every-50-epoch checkpoints kept, final `netG_{a,b}_model.pth` written. **Device:** `--device` instead of a hard-coded `cuda:0`. `torch.autograd.set_detect_anomaly(True)` removed (debug-only slowdown) | none (same networks, losses, 300 + 300 epochs, batch 1, seed 123) |
| `utils/dataset.py` | **File lookup:** `DatasetFromFolder` opened the sketch `b/<name>` with the photo's file name; FS2K style-2 pairs are `a/<stem>.jpg` + `b/<stem>.png`. The sketch is now looked up by stem when the exact name is missing | none where names match (WildSketch, CUHK, CUFSF) |
