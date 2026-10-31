"""Shared generation loop for generate.py and the training val hook (P4 J1–J9, P11, D6, D8, MC15).

- Input: data.sketch_dataset.SketchOnlyDataset only — no photo or reference image reaches generation. With
  `prompt.attributes.enabled` the per-stem attribute prompt is added; those attributes come from the GT photo,
  so such a run is the "Ours + attributes" (witness-description) variant, never the like-for-like row (D6, F10).
- One `inference:` block decides sampler, steps, spacing, CFG, start (noise | xhat) and strength, guess mode,
  control end, precision and batch size; the residual scale is `conditioning.residual_scale`, the same number
  training uses (CN-L1); the prompt is the training prompt; the negative prompt follows the CFG rule (P8).
- Seeds: every image gets torch.Generator(device).manual_seed(hash(seed, stem)) — independent of order and
  batch size (D8, NEW-sdunet-8).
- Output: <stem>.png at the scoring size through utils.writer, then run.json (written last, atomically), so a
  folder without run.json is incomplete. An existing folder is refused unless overwrite=True, which removes its
  PNGs and run.json first and nothing else (MC15).
"""
import hashlib
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from utils.config import plain
from utils.tone import describe as tone_describe
from utils.writer import write_png


def image_seed(seed, stem):
    return int(hashlib.sha256(('%s/%s' % (seed, stem)).encode('utf-8')).hexdigest()[:8], 16) % (2 ** 31)


def negative_prompt_for(cfg):
    p = cfg['prompt']
    return p['negative_style'] if cfg['inference']['guidance_scale'] > p['negative_style_above_cfg'] else p['negative']


def prepare_out_dir(out_dir, overwrite):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise FileExistsError('%s is not empty — choose another run id or pass --overwrite' % out_dir)
        others = [n for n in os.listdir(out_dir) if not (n.endswith('.png') or n == 'run.json')]
        if others:
            raise FileExistsError('%s holds files other than PNGs and run.json: %s' % (out_dir, others[:5]))
        for name in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, name))
    os.makedirs(out_dir, exist_ok=True)


def _write_json_atomic(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_generation(pipe, cfg, dataset, out_dir, seed, extra=None, overwrite=False, log=print, xhat_model=None,
                   device=None):
    """Generate every stem of `dataset` into out_dir. With xhat_model (FrozenCVQVAE with decoder) and pipe None,
    write the frozen-decoder estimate x̂ instead (01_codebook.md M5)."""
    inf = cfg['inference']
    xhat_only = pipe is None
    if xhat_only and xhat_model is None:
        raise ValueError('run_generation needs a pipeline or an x̂ model')
    dev = torch.device(device) if device is not None else pipe.device
    prepare_out_dir(out_dir, overwrite)
    loader = DataLoader(dataset, batch_size=inf['batch_size'], shuffle=False, drop_last=False,
                        num_workers=min(4, cfg['data']['num_workers']), pin_memory=False)
    negative = negative_prompt_for(cfg)
    attribute_prompts = bool(cfg['prompt']['attributes']['enabled'])
    tone_stats = None
    if cfg['tone']['enabled']:
        from utils.tone import load_tone_stats, normalise_tone  # noqa: F401  (used in the x̂ branch)
        tone_stats = pipe.tone_stats if not xhat_only else load_tone_stats(
            cfg['_resolved']['tone_stats_dir'], cfg['tone']['stats_split'])
    if not xhat_only:
        pipe.eval_mode()
        text_cond = pipe.encode_prompt([cfg['prompt']['text']])   # per-batch below when attributes are on
        text_neg = pipe.encode_prompt([negative])
    size_wh = inf['output_size']
    t0 = time.time()
    written = 0
    for batch in loader:
        stems = list(batch['stem'])
        if xhat_only:
            images = xhat_model.estimate(batch['sketch'].to(dev))
            if tone_stats is not None:   # same anchor the SD path would start from
                images = normalise_tone(images.float(), tone_stats, cfg['tone']['method'],
                                        cfg['tone']['strength'], cfg['tone']['target'])
        else:
            generators = [torch.Generator(device=dev).manual_seed(image_seed(seed, s)) for s in stems]
            cond = pipe.encode_prompt(list(batch['prompt'])) if attribute_prompts else text_cond
            images = pipe.generate(batch['sketch'], batch['sketch_raw'], cond, text_neg, generators)
        for image, stem in zip(images, stems):
            write_png(image, out_dir, stem, size_wh)
            written += 1
        log('  %s: %d / %d images' % (os.path.basename(out_dir.rstrip('/')), written, len(dataset)))
    seconds = time.time() - t0
    record = {
        'kind': 'xhat' if xhat_only else 'generation',
        'split': dataset.split, 'images': written, 'complete': bool(dataset.complete),
        'seed': seed, 'per_image_seed': "int(sha256(f'{seed}/{stem}')[:8], 16) % 2**31",
        'output_size_wh': list(size_wh), 'seconds': seconds,
        'seconds_per_image': seconds / max(1, written), 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'prompt': cfg['prompt']['text'], 'negative_prompt': None if xhat_only else negative,
        'prompts': getattr(dataset, 'prompts', None).describe() if getattr(dataset, 'prompts', None) else None,
        'tone': tone_describe(cfg, tone_stats),
        'inference': dict(inf), 'residual_scale': cfg['conditioning']['residual_scale'],
        'conditioning_paths': cfg['conditioning']['paths'],
        'edge_amplification': cfg['conditioning']['edge_amplification'],
        'config': plain(cfg), 'resolved_paths': cfg.get('_resolved'),
    }
    if not xhat_only:
        record['pipeline'] = pipe.info
    if device is not None or not xhat_only:
        if dev.type == 'cuda':
            record['max_mem_gb'] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if extra:
        record['extra'] = extra
    _write_json_atomic(os.path.join(out_dir, 'run.json'), record)
    log('wrote %d PNGs + run.json to %s (%.1f s, %.2f s/image)' % (written, out_dir, seconds, record['seconds_per_image']))
    return {'out_dir': out_dir, 'images': written, 'seconds': seconds}
