"""Aligned identity loss with DECP's IR-SE50 ArcFace (P7, MC8; replaces v1's unaligned whole-image facenet, F1, F2).

- Network: Backbone(112, 50, mode 'ir_se', drop_ratio 0.6) — DECP's IDLoss network; architecture copied from
  1_codebook/utils/id_encoder (models/id_encoder), weights data/pretrained/decp/model_ir_se50.pth with the
  sha256 asserted. Never CurricularFace, the scoring network.
- Crops: 112² similarity-aligned to the harness's ArcFace template using the training GT's precomputed MTCNN
  landmarks; each sample carries its affine_grid matrix (data/sd_dataset.py, mirrored under flip). The same
  matrix crops the prediction and the GT, sampled from their 256² versions.
- Per-row loss 1 − ⟨f(pred crop), f(GT crop)⟩ with the GT embedding detached (DECP's IDLoss form; Backbone
  returns L2-normalised embeddings).
- Pinned to eval mode (`train` overridden), no trainable parameters, float32 with autocast off.
"""
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.id_encoder.model_irse import Backbone
from utils.config import sha256_file
from utils.precision import autocast_disabled

CROP_SIZE = 112


def _load_state(path, hash_verified):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location='cpu')
    except (pickle.UnpicklingError, RuntimeError):
        if not hash_verified:
            raise
        return torch.load(path, map_location='cpu', weights_only=False)


class AlignedIdentityLoss(nn.Module):
    def __init__(self, weights_path, expected_sha256=None):
        super(AlignedIdentityLoss, self).__init__()
        if not os.path.isfile(weights_path):
            raise FileNotFoundError('IR-SE50 weights not found: %s' % weights_path)
        digest = sha256_file(weights_path)
        if expected_sha256 and digest != expected_sha256:
            raise RuntimeError('IR-SE50 sha256 %s != expected %s (%s)' % (digest, expected_sha256, weights_path))
        self.weights_path = os.path.abspath(weights_path)
        self.weights_sha256 = digest
        self.net = Backbone(input_size=CROP_SIZE, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.net.load_state_dict(_load_state(weights_path, bool(expected_sha256)), strict=True)
        for p in self.parameters():
            p.requires_grad_(False)
        super(AlignedIdentityLoss, self).train(False)

    def train(self, mode=True):
        return super(AlignedIdentityLoss, self).train(False)

    @staticmethod
    def crops(images, theta):
        grid = F.affine_grid(theta.float(), [images.shape[0], images.shape[1], CROP_SIZE, CROP_SIZE],
                             align_corners=False)
        return F.grid_sample(images.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, pred, target, theta):
        """pred, target (k, 3, H, W) float32 [-1, 1]; theta (k, 2, 3) → per-row 1 − cos (k,)."""
        with autocast_disabled(pred.device):
            theta = theta.to(pred.device)
            f_pred = self.net(self.crops(pred, theta))
            with torch.no_grad():
                f_gt = self.net(self.crops(target.detach(), theta))
            return 1.0 - (f_pred * f_gt).sum(dim=1)
