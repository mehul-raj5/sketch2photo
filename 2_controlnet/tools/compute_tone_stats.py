"""Tone statistics of the TRAINING photos and of their codebook estimates x̂ (PLAN phase 0).

    python 2_controlnet/tools/compute_tone_stats.py --config 2_controlnet/configs/wildsketch.yaml \
        --split train_dev --device cuda:0

Writes `data/tone_stats/<dataset>/<split>.json`: per-channel CIE-Lab and RGB mean/std plus an L-channel
CDF, for the photos (`photo`) and for x̂ (`xhat`). `utils/tone.py` maps x̂ onto the `photo` statistics at
generation time.

Leakage discipline: `--split train_dev` uses the 356 training stems only and is the file every val-scored run
must use; `--split all` covers the 396 training pairs and belongs to the final test run. **No split reads a
val or test photo** — `training_stems` only ever indexes `train/`.
"""
import argparse
import hashlib
import json
import os
import sys
import time

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from utils import config as C  # noqa: E402  (no torch import)

BINS = 256


def _stats(lab, rgb01):
    """lab (N, 3), rgb01 (N, 3) tensors of pixels -> the JSON block."""
    import torch
    l = lab[:, 0]
    hist = torch.histc(l, bins=BINS, min=0.0, max=100.0)
    cdf = torch.cumsum(hist, dim=0)
    cdf = cdf / cdf[-1].clamp(min=1.0)
    return {
        'lab_mean': [float(v) for v in lab.mean(dim=0)],
        'lab_std': [float(v) for v in lab.std(dim=0)],
        'rgb_mean': [float(v) for v in rgb01.mean(dim=0)],
        'rgb_std': [float(v) for v in rgb01.std(dim=0)],
        'l_cdf': [float(v) for v in cdf],
        'brightness_0_255': float(rgb01.mean() * 255.0),
    }


def main():
    ap = argparse.ArgumentParser(description='Tone statistics of the training photos and their x̂ (phase 0)')
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True, choices=['train_dev', 'all'],
                    help='train_dev = the 356 tuning stems (use for anything scored on val); all = 396 (final run)')
    ap.add_argument('--device', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    C.apply_env(cfg)
    r = cfg['_resolved']
    out_path = args.out or os.path.join(r['tone_stats_dir'], args.split + '.json')
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit('%s exists — pass --overwrite' % out_path)

    import torch
    from data.image_io import index_images, load_rgb, pil_to_tensor, squash
    from data.sd_dataset import training_stems
    from models.cvqvae_encoder import build_frozen_cvqvae
    from utils.tone import rgb_to_lab

    device = torch.device(args.device)
    stems = training_stems(r['dataset_dir'], r['splits_dir'], args.split)
    photos = index_images(os.path.join(r['dataset_dir'], 'train', 'a'))
    sketches = index_images(os.path.join(r['dataset_dir'], 'train', 'b'))
    cvq = build_frozen_cvqvae(cfg, with_decoder=True).to(device)
    size = int(cfg['data']['sketch_size'])

    acc = {'photo': {'lab': [], 'rgb': []}, 'xhat': {'lab': [], 'rgb': []}}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(stems), args.batch):
            chunk = stems[i:i + args.batch]
            photo = torch.stack([pil_to_tensor(squash(load_rgb(photos[s]), size)) for s in chunk]).to(device)
            sketch = torch.stack([pil_to_tensor(squash(load_rgb(sketches[s]), size)) for s in chunk]).to(device)
            xhat = cvq.estimate(sketch)
            for name, image in (('photo', photo), ('xhat', xhat)):
                rgb01 = ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0)
                lab = rgb_to_lab(rgb01)
                acc[name]['rgb'].append(rgb01.permute(0, 2, 3, 1).reshape(-1, 3).cpu())
                acc[name]['lab'].append(lab.permute(0, 2, 3, 1).reshape(-1, 3).cpu())
            if (i + args.batch) % 80 == 0:
                print('%d / %d stems' % (min(i + args.batch, len(stems)), len(stems)), flush=True)

    payload = {'dataset': cfg['run']['dataset'], 'split': args.split, 'n_stems': len(stems),
               'geometry': 'PIL-bicubic squash to %d² (the training geometry, D15)' % size,
               'source': 'training photos only — never val or test (leakage discipline, D5/D6)',
               'codebook': cvq.describe(), 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
               'tool_sha256': C.sha256_file(os.path.abspath(__file__)),
               'command': ' '.join([sys.executable] + sys.argv)}
    for name in ('photo', 'xhat'):
        payload[name] = _stats(torch.cat(acc[name]['lab']), torch.cat(acc[name]['rgb']))
    payload['seconds'] = time.time() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload['sha256'] = hashlib.sha256(body.encode('utf-8')).hexdigest()
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)

    for name in ('photo', 'xhat'):
        s = payload[name]
        print('%-6s brightness %6.1f  Lab mean %s  Lab std %s'
              % (name, s['brightness_0_255'],
                 ['%.1f' % v for v in s['lab_mean']], ['%.1f' % v for v in s['lab_std']]))
    print('wrote %s (%d stems, %.1f s)' % (out_path, len(stems), payload['seconds']))


if __name__ == '__main__':
    main()
