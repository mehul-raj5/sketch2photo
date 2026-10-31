"""Paired training dataset, resumable batch sampler and ID-crop alignment (P1, P3, P7, MC18, D15).

PairedSquashDataset
  - stems from data/splits/<id>/{train_dev,val}.txt, or every paired training stem for split 'all' (D5);
  - photo a/<stem>.* and sketch b/<stem>.* paired by stem, equal sizes asserted at construction (F31);
  - RGB on load; PIL bicubic squash of the original W x H straight to 512² (photo, sketch_raw) and
    256² (sketch) — no pad_and_zoom (F6), no crop, rotation or photo-only jitter (MC18, NEW-pipeline-5);
  - paired horizontal flip decided by hash(seed, epoch, index), so a resumed run replays the same flips
    whatever the worker count (NEW-pipeline-7);
  - no prompt or JSON files are read (F10);
  - with a landmark file: `id_theta`, the affine_grid matrix of the ArcFace 112² crop (P7), mirrored with the
    flip; stems without landmarks fall back to DECP's fixed IDLoss crop.
EpochBatchSampler: batch b of epoch e is a pure function of (seed, e, b); resume starts at any batch.
"""
import hashlib
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from data.image_io import index_images, load_rgb, natural_key, pil_to_tensor, read_stem_list, squash

FLIP_LR = getattr(Image, 'Transpose', Image).FLIP_LEFT_RIGHT

# ArcFace 112² template as the harness aligns (MTCNN.align → get_reference_facial_points(default_square=True)):
# 4_evaluation/decp/evaluate/models/mtcnn/mtcnn_pytorch/src/align_trans.py REFERENCE_FACIAL_POINTS (96 x 112) + (8, 0).
ARCFACE_TEMPLATE_112 = np.array([
    [38.29459953, 51.69630051],   # left eye (image left)
    [73.53179932, 51.50139999],   # right eye
    [56.02519989, 71.73660278],   # nose
    [41.54930115, 92.36550140],   # mouth left
    [70.72990036, 92.20410156],   # mouth right
], dtype=np.float64)
ID_CROP_SIZE = 112
MIRROR_ORDER = [1, 0, 2, 4, 3]
# DECP IDLoss crop x[:, :, 35:223, 32:220] of the 256² image, as a normalised affine_grid box (fallback)
DECP_FIXED_CROP_THETA = np.array([[0.734375, 0.0, -0.015625],
                                  [0.0, 0.734375, 0.0078125]], dtype=np.float32)


def unit_hash(*parts):
    digest = hashlib.sha256('/'.join(str(p) for p in parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little') / float(1 << 64)


def training_stems(dataset_dir, splits_dir, split):
    """Stems of a training split. 'all' = every paired stem of train/; 'train_dev' / 'val' from the split files,
    which must be disjoint and inside train/."""
    photos = index_images(os.path.join(dataset_dir, 'train', 'a'))
    sketches = index_images(os.path.join(dataset_dir, 'train', 'b'))
    unpaired = sorted(set(photos) ^ set(sketches), key=natural_key)
    if unpaired:
        raise ValueError('unpaired training stems in %s: %s' % (dataset_dir, unpaired[:10]))
    paired = sorted(photos, key=natural_key)
    if split == 'all':
        return paired
    if split not in ('train_dev', 'val'):
        raise ValueError('unknown training split %r' % split)
    dev = read_stem_list(os.path.join(splits_dir, 'train_dev.txt'))
    val = read_stem_list(os.path.join(splits_dir, 'val.txt'))
    overlap = sorted(set(dev) & set(val), key=natural_key)
    if overlap:
        raise ValueError('train_dev and val share stems: %s' % overlap[:10])
    stems = dev if split == 'train_dev' else val
    outside = [s for s in stems if s not in photos]
    if outside:
        raise ValueError('split %s has stems outside train/: %s' % (split, outside[:10]))
    return stems


# ----------------------------------------------------------------------------- ID alignment (P7)
def load_landmarks(path):
    """Landmark file written by tools/precompute_landmarks.py."""
    if not os.path.isfile(path):
        raise FileNotFoundError('landmark file not found: %s — run tools/precompute_landmarks.py first '
                                '(or set aux.id.weight 0)' % path)
    with open(path, 'r') as fh:
        data = json.load(fh)
    for key in ('dataset', 'frame', 'landmarks', 'missing'):
        if key not in data:
            raise ValueError('%s: missing key %r' % (path, key))
    points = {}
    for stem, pts in data['landmarks'].items():
        arr = np.asarray(pts, dtype=np.float64)
        if arr.shape != (5, 2) or not np.all(np.isfinite(arr)):
            raise ValueError('%s: bad landmarks for stem %s' % (path, stem))
        points[stem] = arr
    return {'dataset': data['dataset'], 'frame': tuple(int(v) for v in data['frame']),
            'landmarks': points, 'missing': set(data['missing'])}


def similarity_transform(src, dst):
    """Least-squares similarity without reflection (Umeyama) with dst ≈ s·R·src + t; returns a 3 x 3 matrix."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = float(src.shape[0])
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T.dot(sc) / n
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1.0
    rot = u.dot(d).dot(vt)
    var_s = (sc ** 2).sum() / n
    scale = np.trace(np.diag(s).dot(d)) / var_s
    mat = np.eye(3)
    mat[:2, :2] = scale * rot
    mat[:2, 2] = mu_d - scale * rot.dot(mu_s)
    return mat


def arcface_theta(points, frame_wh, flip):
    """(2, 3) float32 affine_grid matrix (align_corners=False) sampling the 112² ArcFace crop from an image in the
    landmark frame. Normalised coordinates make it valid at any resolution of that frame (512² or 256²).
    Landmarks use pixel-centre-at-integer coordinates (MTCNN / cv2 convention)."""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    pts = np.asarray(points, dtype=np.float64).copy()
    if flip:
        pts[:, 0] = (w - 1.0) - pts[:, 0]
        pts = pts[MIRROR_ORDER]
    crop_to_frame = similarity_transform(ARCFACE_TEMPLATE_112, pts)
    c = float(ID_CROP_SIZE)
    norm_to_crop = np.array([[c / 2.0, 0.0, (c - 1.0) / 2.0], [0.0, c / 2.0, (c - 1.0) / 2.0], [0.0, 0.0, 1.0]])
    frame_to_norm = np.array([[2.0 / w, 0.0, 1.0 / w - 1.0], [0.0, 2.0 / h, 1.0 / h - 1.0], [0.0, 0.0, 1.0]])
    return frame_to_norm.dot(crop_to_frame).dot(norm_to_crop)[:2].astype(np.float32)


def fallback_theta(flip):
    theta = DECP_FIXED_CROP_THETA.copy()
    if flip:
        theta[0, 2] = -theta[0, 2]
    return theta


# ----------------------------------------------------------------------------- dataset
class PairedSquashDataset(Dataset):
    def __init__(self, dataset_dir, stems, photo_size=512, sketch_size=256, flip=0.0, seed=0, landmarks=None,
                 prompts=None):
        photo_dir = os.path.join(dataset_dir, 'train', 'a')
        sketch_dir = os.path.join(dataset_dir, 'train', 'b')
        photos = index_images(photo_dir)
        sketches = index_images(sketch_dir)
        self.items = []
        for stem in stems:
            if stem not in photos or stem not in sketches:
                raise ValueError('stem %s has no photo/sketch pair in %s' % (stem, dataset_dir))
            with Image.open(photos[stem]) as p, Image.open(sketches[stem]) as s:
                if p.size != s.size:
                    raise ValueError('pair %s: photo %s vs sketch %s differ in size (F31)' % (stem, p.size, s.size))
            self.items.append((stem, photos[stem], sketches[stem]))
        if not self.items:
            raise ValueError('empty dataset')
        self.photo_size = int(photo_size)
        self.sketch_size = int(sketch_size)
        self.flip = float(flip)
        self.seed = int(seed)
        self.landmarks = landmarks
        self.prompts = prompts
        self.landmarks_found = 0
        if landmarks is not None:
            if landmarks['frame'] != (self.photo_size, self.photo_size):
                raise ValueError('landmark frame %s != photo frame %d²' % (landmarks['frame'], self.photo_size))
            uncovered = [s for s in stems if s not in landmarks['landmarks'] and s not in landmarks['missing']]
            if uncovered:
                raise ValueError('landmark file does not cover %d stems (first: %s)' % (len(uncovered), uncovered[:5]))
            self.landmarks_found = sum(1 for s in stems if s in landmarks['landmarks'])

    @property
    def stems(self):
        return [item[0] for item in self.items]

    def __len__(self):
        return len(self.items)

    def flip_decision(self, index, epoch):
        return self.flip > 0.0 and unit_hash('flip', self.seed, epoch, index) < self.flip

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            index, epoch = int(key[0]), int(key[1])
        else:
            index, epoch = int(key), -1   # -1: evaluation access, flips still hash-determined
        stem, photo_path, sketch_path = self.items[index]
        photo = load_rgb(photo_path)
        sketch = load_rgb(sketch_path)
        flip = self.flip_decision(index, epoch)
        if flip:  # paired: the same transform on photo and sketch, before any resize (MC18)
            photo = photo.transpose(FLIP_LR)
            sketch = sketch.transpose(FLIP_LR)
        sample = {
            'stem': stem,
            'photo': pil_to_tensor(squash(photo, self.photo_size)),         # (3, 512, 512) [-1, 1] VAE + aux target
            'sketch': pil_to_tensor(squash(sketch, self.sketch_size)),      # (3, 256, 256) [-1, 1] CVQ-VAE
            'sketch_raw': pil_to_tensor(squash(sketch, self.photo_size)),   # (3, 512, 512) [-1, 1] edge path
            'flip': bool(flip),
            'prompt': self.prompts.text(stem) if self.prompts is not None else '',
        }
        if self.landmarks is not None:
            pts = self.landmarks['landmarks'].get(stem)
            if pts is None:
                sample['id_theta'] = torch.from_numpy(fallback_theta(flip))
                sample['id_landmarks_found'] = False
            else:
                sample['id_theta'] = torch.from_numpy(arcface_theta(pts, self.landmarks['frame'], flip))
                sample['id_landmarks_found'] = True
        return sample


class EpochBatchSampler(Sampler):
    """drop_last batches of (index, epoch) keys; the permutation of epoch e is seeded by (seed, e)."""

    def __init__(self, n, batch_size, seed, epoch, start_batch=0):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    @staticmethod
    def batches_per_epoch(n, batch_size):
        return int(n) // int(batch_size)

    def __len__(self):
        return max(0, self.batches_per_epoch(self.n, self.batch_size) - self.start_batch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000003 + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        for b in range(self.start_batch, self.batches_per_epoch(self.n, self.batch_size)):
            chunk = perm[b * self.batch_size:(b + 1) * self.batch_size]
            yield [(i, self.epoch) for i in chunk]


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    base = int(info.seed) % (2 ** 32) if info is not None else 0
    np.random.seed(base)
    random.seed(base)


def make_train_loader(dataset, batch_size, seed, epoch, start_batch, num_workers, pin_memory):
    sampler = EpochBatchSampler(len(dataset), batch_size, seed, epoch, start_batch)
    g = torch.Generator()
    g.manual_seed(seed * 7919 + epoch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker, generator=g)


class ContinuousBatchSampler(Sampler):
    """Endless EpochBatchSampler stream from (start_epoch, start_batch): the same batches in the same order, so one
    DataLoader (and its workers) serves a whole session while the loop counts the position exactly."""

    def __init__(self, n, batch_size, seed, start_epoch, start_batch=0):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_epoch = int(start_epoch)
        self.start_batch = int(start_batch)
        if EpochBatchSampler.batches_per_epoch(self.n, self.batch_size) < 1:
            raise ValueError('dataset of %d samples is smaller than one batch of %d' % (self.n, self.batch_size))

    def __iter__(self):
        epoch, first = self.start_epoch, self.start_batch
        while True:
            for keys in EpochBatchSampler(self.n, self.batch_size, self.seed, epoch, first):
                yield keys
            epoch += 1
            first = 0


def make_continuous_train_loader(dataset, batch_size, seed, start_epoch, start_batch, num_workers, pin_memory):
    sampler = ContinuousBatchSampler(len(dataset), batch_size, seed, start_epoch, start_batch)
    g = torch.Generator()
    g.manual_seed(seed * 7919 + start_epoch * 131 + start_batch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                      worker_init_fn=seed_worker, generator=g)


def make_eval_loader(dataset, batch_size, num_workers, pin_memory, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                      pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)
