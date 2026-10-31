"""Frozen CVQ-VAE of DECP stage 1 (photo domain): sketch tokens and the frozen-decoder estimate x̂.

CB-L1 (D4, F35, NEW-codebook-1…5, NEW-codebook-7):
- `train(mode)` is overridden, so the module never leaves eval mode — in training mode
  `VectorQuantiser` rewrites its codes in place and consumes the CUDA RNG;
- every parameter has requires_grad False; `tokens()` / `estimate()` run under no_grad, in float32,
  with autocast disabled, in training and in inference alike (fp16 token flips, NEW-codebook-2);
- the checkpoint file sha256 and the sha256 of `codebook.embedding.weight` are asserted at build when the
  config provides them; `integrity()` recomputes the codebook tensor hashes for every checkpoint;
- each of the loaded groups must exist and load strictly (a missing group raises, NEW-codebook-3);
- the checkpoint path is derived from `run.dataset` by utils/config.py (`<id>_cvq`, NEW-codebook-4).
CB-L2 / CB §8e: with `with_decoder=True`, x̂ = D(post_quant_conv(z_q)) — a pixel-aligned photo estimate
from the sketch tokens and training-set weights only (D6-clean), used by the `inference.start: xhat` option.
`utils/modules.py` and `utils/CVQ_quantise.py` are DECP's code, kept numerically identical (F28).
"""
import contextlib
import hashlib
import os
import sys

import torch
import torch.nn as nn

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2_controlnet (NEW-codebook-7)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from utils.CVQ_quantise import VectorQuantiser  # noqa: E402
from utils.config import sha256_file  # noqa: E402
from utils.modules import Decoder, Encoder  # noqa: E402


def tensor_sha256(tensor):
    """sha256 of the raw float32 little-endian bytes (01_codebook.md §5 convention)."""
    arr = tensor.detach().to(dtype=torch.float32, device='cpu').contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def autocast_disabled(device):
    """Context that switches autocast off for `device` (no-op context for device types without autocast)."""
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except (RuntimeError, ValueError):
        return contextlib.nullcontext()


def _torch_load_tensors(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only; the file hash was checked before
        return torch.load(path, map_location='cpu')


class FrozenCVQVAE(nn.Module):
    """sketch (B, 3, 256, 256) [-1, 1] → z_q (B, 256, 32, 32) float32; optionally → x̂ (B, 3, 256, 256) [-1, 1]."""

    def __init__(self, weights_path, num_embeddings=1024, embed_dim=256, with_decoder=False,
                 expected_file_sha256=None, expected_embedding_sha256=None):
        super(FrozenCVQVAE, self).__init__()
        self.encoder = Encoder(input_channel=3, output_channel=embed_dim)
        self.quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.codebook = VectorQuantiser(num_embeddings, embed_dim, 0.25, contras_loss=False)
        self.with_decoder = bool(with_decoder)
        if self.with_decoder:
            self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
            self.decoder = Decoder(input_channel=embed_dim)
        self.weights_path = os.path.abspath(weights_path)
        self.file_sha256 = self._load(self.weights_path, expected_file_sha256)
        for p in self.parameters():
            p.requires_grad_(False)
        super(FrozenCVQVAE, self).train(False)
        self.embedding_sha256 = tensor_sha256(self.codebook.embedding.weight)
        self.embed_prob_sha256 = tensor_sha256(self.codebook.embed_prob)
        if expected_embedding_sha256 and self.embedding_sha256 != expected_embedding_sha256:
            raise RuntimeError('codebook.embedding.weight sha256 %s != expected %s (%s) — wrong codebook?'
                               % (self.embedding_sha256, expected_embedding_sha256, self.weights_path))
        self.hashes_asserted = {'file': bool(expected_file_sha256), 'embedding': bool(expected_embedding_sha256)}

    # -- D4: never leave eval mode ------------------------------------------------------------
    def train(self, mode=True):
        return super(FrozenCVQVAE, self).train(False)

    def _check_eval(self):
        if self.training or self.codebook.training:
            raise RuntimeError('FrozenCVQVAE left eval mode — the quantiser would rewrite its codes (F35, D4)')

    # -- loading --------------------------------------------------------------------------------
    def _load(self, path, expected_file_sha256):
        if not os.path.isfile(path):
            raise FileNotFoundError('codebook checkpoint not found: %s' % path)
        digest = sha256_file(path)
        if expected_file_sha256 and digest != expected_file_sha256:
            raise RuntimeError('codebook file sha256 %s != expected %s (%s)' % (digest, expected_file_sha256, path))
        state = _torch_load_tensors(path)
        groups = [('encoder.', self.encoder), ('quant_conv.', self.quant_conv), ('codebook.', self.codebook)]
        if self.with_decoder:
            groups += [('post_quant_conv.', self.post_quant_conv), ('decoder.', self.decoder)]
        for prefix, module in groups:
            sub = dict((k[len(prefix):], v) for k, v in state.items() if k.startswith(prefix))
            if not sub:
                raise RuntimeError('%s has no "%s*" tensors (NEW-codebook-3)' % (path, prefix))
            module.load_state_dict(sub, strict=True)
        return digest

    # -- forward paths --------------------------------------------------------------------------
    @torch.no_grad()
    def tokens(self, sketch):
        """Quantised token grid in float32, computed outside autocast (CB-L1)."""
        self._check_eval()
        with autocast_disabled(sketch.device):
            h, _ = self.encoder(sketch.float())
            h = self.quant_conv(h)
            z_q, _, _ = self.codebook(h)
        return z_q.float()

    def forward(self, sketch):
        return self.tokens(sketch)

    @torch.no_grad()
    def estimate(self, sketch):
        """Frozen-decoder estimate x̂ = D(post_quant_conv(z_q)), clamped to [-1, 1] (CB §8e)."""
        if not self.with_decoder:
            raise RuntimeError('FrozenCVQVAE was built without the decoder (with_decoder=False)')
        z_q = self.tokens(sketch)
        with autocast_disabled(sketch.device):
            x = self.decoder(self.post_quant_conv(z_q))
        return x.float().clamp(-1.0, 1.0)

    # -- integrity ------------------------------------------------------------------------------
    def integrity(self):
        emb = tensor_sha256(self.codebook.embedding.weight)
        prob = tensor_sha256(self.codebook.embed_prob)
        ok = (emb == self.embedding_sha256 and prob == self.embed_prob_sha256
              and not self.training and not self.codebook.training)
        return {'embedding_sha256': emb, 'embed_prob_sha256': prob, 'eval_mode': not self.codebook.training,
                'ok': bool(ok)}

    def assert_integrity(self):
        info = self.integrity()
        if not info['ok']:
            raise RuntimeError('frozen codebook changed or left eval mode (F35, D4): %s' % info)
        return info

    def describe(self):
        return {'weights_path': self.weights_path, 'file_sha256': self.file_sha256,
                'embedding_sha256': self.embedding_sha256, 'embed_prob_sha256': self.embed_prob_sha256,
                'hashes_asserted': self.hashes_asserted, 'with_decoder': self.with_decoder}


def build_frozen_cvqvae(cfg, with_decoder=False):
    """Build from a validated config: `<data_root>/codebooks/<run.dataset>_cvq/netG_a_model.pth`."""
    cb = cfg['codebook']
    return FrozenCVQVAE(cfg['_resolved']['codebook_file'], num_embeddings=cb['num_embeddings'],
                        embed_dim=cb['embed_dim'], with_decoder=with_decoder,
                        expected_file_sha256=cb['sha256_file'], expected_embedding_sha256=cb['sha256_embedding'])
