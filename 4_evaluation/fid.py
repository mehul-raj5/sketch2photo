"""FID as DECP's recovered calculate.py (pytorch_fid.calculate_fid_given_paths, batch 32, dims 2048) and KID on the
same Inception pool3 features.

The activations are computed once per folder with pytorch_fid's own functions, so FID equals
calculate_fid_given_paths([pred, gt], 32, device, 2048) — checked by the self-test.
"""
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance, get_activations
from pytorch_fid.inception import InceptionV3

import protocol as P


class InceptionFeatures(object):
    def __init__(self, device, num_workers=4):
        self.device = device
        self.num_workers = num_workers
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[P.FID_DIMS]
        self.model = InceptionV3([block_idx]).to(device).eval()

    @torch.no_grad()
    def __call__(self, files):
        return get_activations(list(files), self.model, batch_size=P.FID_BATCH, dims=P.FID_DIMS,
                               device=self.device, num_workers=self.num_workers)


def fid_from_features(f1, f2):
    mu1, s1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
    mu2, s2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def _mmd2_poly(x, y):
    """Unbiased MMD^2 with the kernel (x.y/d + 1)^3 (clean-fid's kernel_distance for one subset)."""
    d = x.shape[1]
    m = x.shape[0]
    a = (x @ x.T / d + 1) ** 3 + (y @ y.T / d + 1) ** 3
    b = (x @ y.T / d + 1) ** 3
    return float(((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m) / m)


def kid_from_features(f_pred, f_gt, n_subsets=P.KID_SUBSETS, subset_size=P.KID_SUBSET_SIZE, seed=P.KID_SEED):
    f_pred = np.asarray(f_pred, dtype=np.float64)
    f_gt = np.asarray(f_gt, dtype=np.float64)
    m = min(len(f_pred), len(f_gt), subset_size)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_subsets):
        x = f_gt[rng.choice(len(f_gt), m, replace=False)]
        y = f_pred[rng.choice(len(f_pred), m, replace=False)]
        vals.append(_mmd2_poly(x, y))
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'subset_size': m, 'subsets': n_subsets}


def kid_paired_subsets(f_pred, f_gt, size, count, seed=P.KID_SEED):
    """KID over `count` random subsets of `size` stems, the same stems on both sides (the val-size reference, Q5)."""
    f_pred = np.asarray(f_pred, dtype=np.float64)
    f_gt = np.asarray(f_gt, dtype=np.float64)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(count):
        idx = rng.choice(len(f_gt), size, replace=False)
        vals.append(_mmd2_poly(f_gt[idx], f_pred[idx]))
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'subset_size': size, 'subsets': count}
