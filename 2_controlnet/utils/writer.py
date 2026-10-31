"""The v2 output writer (P4 J8, MC2, MC3, MC15; replaces utils/sd_utils.save_image, NEW-pipeline-10).

(3, H, W) float in [-1, 1] → (x + 1) / 2 · 255 → clip → uint8 by truncation (DECP `save_img` parity)
→ PIL image → PIL bicubic to the dataset's scoring size (256 × 256 for wildsketch / fs2k; 200 × 250 for
cuhk / color_feret / cufsf) — the same operator the harness applies to the GT → 8-bit RGB PNG `<stem>.png`.
Never JPEG, never any other resampler, never a post-hoc filter (MC5).
"""
import os

import numpy as np
from PIL import Image

BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC


def to_uint8_hwc(image):
    arr = image.detach().float().cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError('expected a (3, H, W) image, got %s' % (arr.shape,))
    arr = (np.transpose(arr, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def write_png(image, out_dir, stem, size_wh):
    """Write one prediction as <out_dir>/<stem>.png at size_wh = (W, H); returns the path."""
    pil = Image.fromarray(to_uint8_hwc(image), mode='RGB')
    size_wh = (int(size_wh[0]), int(size_wh[1]))
    if pil.size != size_wh:
        pil = pil.resize(size_wh, BICUBIC)
    path = os.path.join(out_dir, stem + '.png')
    pil.save(path, format='PNG')
    return path
