# data/ (not in git)

```
data/
├── datasets/<id>/{train,test}/{a,b}/   built by 0_data_preparation/
├── splits/<id>/{train_dev,val}.txt     0_data_preparation/make_splits.py
├── sources/cufsf_cuhk/                 CUFSF sketch downloads (only for build_cufsf.py)
├── codebooks/<id>_cvq/                 Drive `codebooks/` (or 1_codebook/)
├── tone_stats/<id>/all.json            Drive `tone_stats/` (or 2_controlnet/tools/compute_tone_stats.py)
└── pretrained/                         Drive `frozen_models/`
```
