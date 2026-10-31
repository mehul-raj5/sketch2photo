# checkpoints/ (not in git)

`checkpoints/<dataset>/step_0014000/` — the released ControlNet + SketchAdapter checkpoint of each dataset, from the Drive folder
(`scripts/link_drive.sh`). Files: `trainable.pt` (raw weights), `ema.pt` (EMA weights, used at inference), `config_resolved.yaml`
(the training config), `meta.json` (code / input hashes and environment of the run), `trainer_state.pt`.
