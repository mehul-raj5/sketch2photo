"""Image I/O shared by the training and generation datasets (P1, D15, MC2; F36, F37).

Nothing in this module knows about photos or sketches: it lists image files by stem, loads them as RGB,
squashes them with PIL bicubic (DECP's `load_img` geometry) and converts them to [-1, 1] tensors.
"""
import os

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC


def natural_key(stem):
    return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)


def index_images(folder):
    """{stem: path} for the image files of one folder; two files with the same stem raise."""
    if not os.path.isdir(folder):
        raise FileNotFoundError('image folder not found: %s' % folder)
    out = {}
    for name in os.listdir(folder):
        if name.startswith('.'):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if stem in out:
            raise ValueError('two images share the stem %r in %s' % (stem, folder))
        out[stem] = os.path.join(folder, name)
    return out


def read_stem_list(path):
    """One stem per line; empty lines ignored; duplicates raise."""
    if not os.path.isfile(path):
        raise FileNotFoundError('split file not found: %s' % path)
    stems = []
    with open(path, 'r') as fh:
        for line in fh:
            stem = line.strip()
            if stem:
                stems.append(stem)
    if not stems:
        raise ValueError('split file is empty: %s' % path)
    if len(set(stems)) != len(stems):
        raise ValueError('duplicate stems in %s' % path)
    return stems


def load_rgb(path):
    """Every image is RGB on load: CUHK sketches are grayscale, some FS2K sketches RGBA (F36, F37)."""
    with Image.open(path) as im:
        return im.convert('RGB')


def squash(img, size):
    """D15 squash: PIL bicubic of the original W x H to size x size, aspect ratio not preserved."""
    return img.resize((int(size), int(size)), BICUBIC)


def pil_to_tensor(img):
    """uint8 RGB PIL image -> float32 (3, H, W) in [-1, 1], equal to ToTensor + Normalize(0.5, 0.5)."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError('expected an RGB image, got array shape %s' % (arr.shape,))
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))) / 127.5 - 1.0
