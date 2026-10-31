"""ours generation — the single generator for val sweeps and final test runs (P4, J1–J9; D5, D6, D8, MC15).

Val (tuning):   python generate.py --checkpoint <outputs>/wildsketch/checkpoints/<run_id> --split val --device cuda:0
                [--set inference.guidance_scale=1.0 --set conditioning.residual_scale=1.1 ...] [--run_id NAME]
Test (final):   python generate.py --checkpoint <.../step_0010000> --split test --seeds 0 1 2 --final --device cuda:0
x̂ only (M5):    python generate.py --xhat_only --config 2_controlnet/configs/wildsketch.yaml --split val --device cuda:0

- Reads the run's resolved config from the checkpoint (19 channels, <id>_cvq, prompt, conditioning); only the
  inference block, the prompt texts, `conditioning.residual_scale` and `conditioning.paths` may be overridden,
  and every override is recorded in run.json.
- Loads the EMA weights by default (`--raw_weights` for the M10 comparison).
- Sketches only (data.sketch_dataset); per-image seeds; <outputs>/<id>/<split>/<run_id>/<stem>.png at the
  scoring size + run.json with checkpoint / EMA / code sha256. Several seeds → one folder per seed (<run_id>_seed<s>).
"""
import argparse
import os
import sys

CODE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '2_controlnet')  # shared model code
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from utils import config as C  # noqa: E402  (no torch import)
from data.prompts import Prompts  # noqa: E402
from utils.checkpoint import resolve_checkpoint  # noqa: E402  (torch imported lazily)

GEN_OVERRIDABLE = ('inference', 'prompt.text', 'prompt.negative', 'prompt.negative_style',
                   'prompt.negative_style_above_cfg', 'prompt.attributes', 'conditioning.residual_scale',
                   'conditioning.paths', 'tone', 'model.vae_decoder')


def parse_args():
    p = argparse.ArgumentParser(description='ours generation')
    p.add_argument('--checkpoint', help='run checkpoint folder (newest checkpoint) or one step_XXXXXXX folder')
    p.add_argument('--config', help='YAML config — only with --xhat_only and no checkpoint')
    p.add_argument('--split', required=True, choices=('val', 'train_dev', 'test'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0], help='one output folder per seed (D8: 0 1 2 for test)')
    p.add_argument('--run_id', default=None, help='output run id (default: <train run id>_step<N>)')
    p.add_argument('--device', required=True, help='torch device, e.g. cuda:0 — never defaulted (D12)')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='override of inference.*, prompt texts, conditioning.residual_scale / paths; repeatable')
    p.add_argument('--raw_weights', action='store_true', help='use the raw instead of the EMA weights')
    p.add_argument('--xhat_only', action='store_true', help='write the frozen-decoder estimate x̂ (needs no SD model)')
    p.add_argument('--limit', type=int, default=None, help='first N stems only (smoke tests; run.json complete=false)')
    p.add_argument('--overwrite', action='store_true', help='replace the PNGs and run.json of an existing folder')
    p.add_argument('--sketch_dir', default=None, help='folder of (degraded) sketches <stem>.<ext> that replaces the split\'s sketches (robustness study); same stems required')
    p.add_argument('--final', action='store_true', help='required for --split test: test is for final numbers only (D5)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.split == 'test' and not args.final:
        raise SystemExit('--split test is for final runs only (D5, rule 5); add --final to confirm')
    ckpt = None
    if args.checkpoint:
        ckpt = resolve_checkpoint(args.checkpoint, inference=True)
        cfg_path = os.path.join(ckpt, 'config_resolved.yaml')
        train_cfg = C.load_config(cfg_path)
        cfg = C.load_config(cfg_path, args.overrides, allowed_prefixes=GEN_OVERRIDABLE)
    elif args.xhat_only and args.config:
        train_cfg = None
        cfg = C.load_config(args.config, args.overrides, allowed_prefixes=GEN_OVERRIDABLE)
    else:
        raise SystemExit('pass --checkpoint (or --xhat_only with --config)')
    if args.raw_weights:
        cfg['inference']['use_ema'] = False
    C.apply_env(cfg)

    import torch
    from data.sketch_dataset import SketchOnlyDataset
    from utils.generation import run_generation

    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    r = cfg['_resolved']
    prompts = Prompts(cfg, args.split, r)
    dataset = SketchOnlyDataset(r['dataset_dir'], r['splits_dir'], args.split, cfg['data']['photo_size'],
                                cfg['data']['sketch_size'], limit=args.limit, prompts=prompts)
    if args.sketch_dir:
        from data.image_io import index_images
        alt = index_images(args.sketch_dir)
        lack = [st for st in dataset.stems if st not in alt]
        if lack:
            raise SystemExit('--sketch_dir lacks %d stems (first: %s)' % (len(lack), lack[:5]))
        dataset.paths = [alt[st] for st in dataset.stems]
    extra = {'argv': sys.argv, 'overrides': r.get('overrides'), 'code_sha256': C.code_sha256()}
    if train_cfg is not None:
        extra['changed_from_training_config'] = C.config_diff(C.plain(train_cfg), C.plain(cfg))

    if args.xhat_only:
        from models.cvqvae_encoder import build_frozen_cvqvae
        model = build_frozen_cvqvae(cfg, with_decoder=True).to(device)
        extra['codebook'] = model.describe()
        run_id = args.run_id or 'xhat_%s' % cfg['run']['dataset']
        pipe = None
    else:
        from models.controlnet_pipeline import SketchToPhotoPipeline
        from utils.checkpoint import load_part
        from utils.ema import EMA
        pipe = SketchToPhotoPipeline(cfg, device, with_decoder=cfg['inference']['start'] == 'xhat')
        trainable = load_part(ckpt, 'trainable')
        pipe.controlnet.load_state_dict(trainable['controlnet'], strict=True)
        pipe.adapter.load_state_dict(trainable['adapter'], strict=True)
        del trainable
        extra['checkpoint'] = ckpt
        extra['trainable_sha256'] = C.sha256_file(os.path.join(ckpt, 'trainable.pt'))
        if cfg['inference']['use_ema']:
            ema_updates = EMA.copy_state_into(pipe.trainable_modules(), load_part(ckpt, 'ema'))
            extra['weights'] = 'ema'
            extra['ema_updates'] = ema_updates
            extra['ema_sha256'] = C.sha256_file(os.path.join(ckpt, 'ema.pt'))
        else:
            extra['weights'] = 'raw'
        pipe.eval_mode()
        model = None
        step = os.path.basename(ckpt).split('_')[-1].lstrip('0') or '0'
        run_id = args.run_id or '%s_step%s' % (cfg['run']['run_id'], step)
    if not C._run_id(run_id):
        raise SystemExit('invalid run id %r' % run_id)

    base = os.path.join(r['dataset_outputs'], args.split)
    for seed in args.seeds:
        name = run_id if len(args.seeds) == 1 else '%s_seed%d' % (run_id, seed)
        run_generation(pipe, cfg, dataset, os.path.join(base, name), seed, extra=extra, overwrite=args.overwrite,
                       xhat_model=model, device=device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
