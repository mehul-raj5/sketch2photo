"""Tone normalisation of the frozen-decoder estimate x̂ (PLAN phase 0, 2026-09-22).

Tokenising a mostly-white sketch with the photo codebook gives a pale anchor: measured on WildSketch test,
x̂ has brightness 178 / saturation 30 against 121 / 85 for the real photos, and SDEdit from x̂ inherits that
tone almost exactly (178 / 31). This module maps x̂ towards the tone of the **training** photos.

Leakage: the per-image source statistics come from x̂ itself, which derives only from the sketch; the target
statistics are a handful of numbers computed once over the training split (`tools/compute_tone_stats.py`).
It is a dataset-level prior from the train split, not DECP's per-image AdaIN from a reference photo (F12).
"""
import json
import os

import torch

METHODS = ('lab_moments', 'rgb_moments', 'hist_l')

# sRGB <-> CIE-Lab (D65). Matrices as row-vectors times columns, applied to (B, 3, H, W) in [0, 1].
_RGB2XYZ = ((0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041))
_XYZ2RGB = ((3.2404542, -1.5371385, -0.4985314),
            (-0.9692660, 1.8760108, 0.0415560),
            (0.0556434, -0.2040259, 1.0572252))
_WHITE = (0.950489, 1.0, 1.088840)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _matmul(x, m):
    r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([m[i][0] * r[0] + m[i][1] * r[1] + m[i][2] * r[2] for i in range(3)], dim=1)


def rgb_to_lab(rgb01):
    lin = torch.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    xyz = _matmul(lin, _RGB2XYZ)
    xyz = torch.cat([xyz[:, i:i + 1] / _WHITE[i] for i in range(3)], dim=1)
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-8) ** (1.0 / 3.0), (_KAPPA * xyz + 16.0) / 116.0)
    return torch.cat([116.0 * f[:, 1:2] - 16.0, 500.0 * (f[:, 0:1] - f[:, 1:2]), 200.0 * (f[:, 1:2] - f[:, 2:3])], 1)


def lab_to_rgb(lab):
    fy = (lab[:, 0:1] + 16.0) / 116.0
    fx = fy + lab[:, 1:2] / 500.0
    fz = fy - lab[:, 2:3] / 200.0
    f = torch.cat([fx, fy, fz], dim=1)
    f3 = f ** 3
    xyz = torch.where(f3 > _EPS, f3, (116.0 * f - 16.0) / _KAPPA)
    xyz = torch.cat([xyz[:, i:i + 1] * _WHITE[i] for i in range(3)], dim=1)
    lin = _matmul(xyz, _XYZ2RGB).clamp(min=0.0)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / 2.4) - 0.055)


def load_tone_stats(directory, split):
    path = os.path.join(directory, split + '.json')
    if not os.path.isfile(path):
        raise FileNotFoundError('tone statistics %s not found — run tools/compute_tone_stats.py --split %s'
                                % (path, split))
    with open(path) as fh:
        payload = json.load(fh)
    payload['path'] = path
    return payload


def _moments(x, dims):
    return x.mean(dim=dims, keepdim=True), x.std(dim=dims, keepdim=True).clamp(min=1e-5)


def _match_moments(x, target_mean, target_std, strength):
    """Per-image moment matching towards (target_mean, target_std), blended by `strength`."""
    mean, std = _moments(x, (2, 3))
    tgt_mean = torch.tensor(target_mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    tgt_std = torch.tensor(target_std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    matched = (x - mean) / std * tgt_std + tgt_mean
    return x + strength * (matched - x)


def _match_hist_l(l, cdf, strength):
    """Histogram-match the L channel (0..100) onto the target CDF (256 bins), blended by `strength`."""
    target = torch.tensor(cdf, dtype=torch.float32, device=l.device)           # (256,) cumulative, 0..1
    levels = torch.linspace(0.0, 100.0, target.numel(), device=l.device)
    out = torch.empty_like(l)
    for i in range(l.shape[0]):
        flat = l[i].reshape(-1)
        order = torch.argsort(flat)
        ranks = torch.empty_like(flat)
        ranks[order] = torch.arange(flat.numel(), device=l.device, dtype=flat.dtype) / max(1, flat.numel() - 1)
        idx = torch.searchsorted(target.contiguous(), ranks.clamp(0.0, 1.0).contiguous())
        out[i] = levels[idx.clamp(max=target.numel() - 1)].reshape(l.shape[1:])
    return l + strength * (out - l)


def normalise_tone(images, stats, method='lab_moments', strength=1.0, target='photo'):
    """(B, 3, H, W) in [-1, 1] -> tone-normalised copy in [-1, 1]. fp32, autocast-free, deterministic."""
    if method not in METHODS:
        raise ValueError('unknown tone method %r' % method)
    if strength <= 0.0:
        return images
    tgt = stats[target]
    x = ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    if method == 'rgb_moments':
        x = _match_moments(x, tgt['rgb_mean'], tgt['rgb_std'], strength)
    else:
        lab = rgb_to_lab(x)
        if method == 'lab_moments':
            lab = _match_moments(lab, tgt['lab_mean'], tgt['lab_std'], strength)
        else:  # hist_l: histogram-match L, moment-match a/b
            l = _match_hist_l(lab[:, 0:1], tgt['l_cdf'], strength)
            ab = _match_moments(lab[:, 1:3], tgt['lab_mean'][1:], tgt['lab_std'][1:], strength)
            lab = torch.cat([l, ab], dim=1)
        x = lab_to_rgb(lab)
    return (x.clamp(0.0, 1.0) * 2.0 - 1.0)


def describe(cfg, stats):
    t = cfg['tone']
    return {'enabled': bool(t['enabled']), 'method': t['method'], 'strength': t['strength'],
            'stats_split': t['stats_split'], 'target': t['target'],
            'stats_file': stats.get('path') if stats else None,
            'stats_sha256': stats.get('sha256') if stats else None}
