# 0_data_preparation

No dataset images are distributed. Each builder writes `data/datasets/<id>/{train,test}/{a,b}/<stem>.<ext>` (a = photo, b = sketch),
copying the downloaded files byte for byte (renamed, never re-encoded). Images are resized at load time.

| Script | Builds |
|---|---|
| `prepare_cuhk.py --photos <dir> --sketches <dir>` | `cuhk`: CUFS student subset, 88 / 100 (sketches `F2-*`/`M2-*` = train) |
| `prepare_fs2k.py --fs2k <FS2K root>` | `fs2k`: official `anno_train.json` / `anno_test.json` split, 1 058 / 1 046 |
| `build_cufsf.py --variant gray --feret <colorferet root> --apply` | `cufsf`: CUFSF artist sketches + the original gray FERET photos, eyes aligned to CUHK's crop framing at 200 × 250, first 250 subjects = train, 944 test. Needs the CUFSF sketch files from CUHK MMLab in `data/sources/cufsf_cuhk/` (`original_sketch.zip`, `cropped_sketch.zip`, `sketch_points.zip`, `photo_points.zip`, `feret_filenames.txt`) and the NIST Color FERET release (licensed by NIST; do not redistribute the result) |
| — | `wildsketch`: the official release already has the 396 / 400 split; put photos in `a/` and sketches in `b/` with their released stems `1`–`800` |
| `make_splits.py --dataset <id> --n-val <n> --seed 0` | `data/splits/<id>/{train_dev,val}.txt` for tuning (val is used only for choosing inference settings) |

sha256 of the splits used in the paper (to check `make_splits.py` output):
```
4554efe0ee0290786f85e3ff43867fde1299226fbdfc4eed3ea0d90dd1aaf471  data/splits/wildsketch/train_dev.txt
42b7f71bf0b6c3b2d3abc0619958c0b37c9a5c959d8352a36505df803324b7bf  data/splits/wildsketch/val.txt
2576a7eb0c45d9eff452b0b675158884b4431607594f833b659caa88287c0930  data/splits/cufsf/train_dev.txt
0256ee42ac45d1099e46b8ce51c7690708c91bfe1eb3a19617b8c7dfa0e7cca1  data/splits/cufsf/val.txt
3db1abadbf7b7f58764c21657b9ee7abe5d4c6fb281ff3784f2429e8680c360e  data/splits/cuhk/train_dev.txt
d31fdb1efabb8dbadfa013ed38a26c72a62c7a9407d15b99c0fdc18a78755c9b  data/splits/cuhk/val.txt
7ec1f37a7592ba3d7facfb5d437b63f496a88ef440a899268048da3c8f7639d9  data/splits/fs2k/train_dev.txt
9080654ecef75f4be62f831e488a9b23603193256837e2c081cf76321a308869  data/splits/fs2k/val.txt
```
