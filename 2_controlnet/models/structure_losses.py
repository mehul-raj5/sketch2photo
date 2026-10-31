"""Auxiliary structure losses at the scoring scale (P6, P12, MC11, MC12; replaces feature_matching_loss.py).

Inputs: decoded x̂₀ and the photo, both downsampled to 256² (antialiased bicubic), float32, [-1, 1].
Per-sample terms (reduction 'none'); the pipeline applies the timestep masks and min(1, SNR) weights:
- lpips   : LPIPS-VGG (lpips v0.1) on [-1, 1]                                   — aux.lpips_vgg
- scharr  : L1 of Scharr magnitudes + 0.5 · L1 of the x / y components, gray      — aux.scharr
- gmsd    : piq.GMSDLoss on gray [0, 1]                                         — aux.gmsd
- fsim    : piq.FSIMLoss(chromatic=False) on gray [0, 1], rows t ≤ its max_t      — aux.fsim_surrogate (flag, OFF;
            if used, the paper must report it as a training loss, README rule 8)
- ms_ssim : piq.MultiScaleSSIMLoss on gray [0, 1], rows t ≤ its max_t            — aux.ms_ssim (flag, OFF)
Gray uses MATLAB rgb2gray weights, as FSIM is scored (MC12). A configured term whose library is missing
raises (no silent disable, NEW-pipeline-6). Loss networks stay in eval mode and have no trainable parameters;
all terms are computed in float32 with autocast off.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.precision import autocast_disabled

RGB2GRAY = (0.2989, 0.5870, 0.1140)
TEXTURE_CUTOFF = 32   # radius in cycles per 256-px image: only the upper three quarters of the band (fine texture)


def resize_antialiased(x, size):
    """(B, C, H, W) → (B, C, size, size) float32, antialiased bicubic, clamped to [-1, 1]."""
    x = x.float()
    if x.shape[-2] == size and x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False, antialias=True).clamp(-1.0, 1.0)


def to_gray01(x):
    x01 = ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return x01[:, 0:1] * RGB2GRAY[0] + x01[:, 1:2] * RGB2GRAY[1] + x01[:, 2:3] * RGB2GRAY[2]


class StructureLosses(nn.Module):
    def __init__(self, aux_cfg):
        super(StructureLosses, self).__init__()
        fsim_cfg, msssim_cfg = aux_cfg['fsim_surrogate'], aux_cfg['ms_ssim']
        self.weights = {
            'lpips': float(aux_cfg['lpips_vgg']),
            'scharr': float(aux_cfg['scharr']),
            'gmsd': float(aux_cfg['gmsd']),
            'fsim': float(fsim_cfg['weight']) if fsim_cfg['enabled'] else 0.0,
            'ms_ssim': float(msssim_cfg['weight']) if msssim_cfg['enabled'] else 0.0,
        }
        tex_cfg = aux_cfg.get('texture') or {'enabled': False, 'max_t': 300, 'weight': 0.0}
        self.weights['texture'] = float(tex_cfg['weight']) if tex_cfg['enabled'] else 0.0
        self._radial = {}
        self.max_t = {'fsim': int(fsim_cfg['max_t']), 'ms_ssim': int(msssim_cfg['max_t']), 'texture': int(tex_cfg['max_t'])}
        self.lpips = None
        if self.weights['lpips'] > 0:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError('aux.lpips_vgg > 0 needs the `lpips` package') from exc
            self.lpips = lpips.LPIPS(net='vgg', verbose=False)
        self.gmsd = self.fsim = self.ms_ssim = None
        if any(self.weights[k] > 0 for k in ('gmsd', 'fsim', 'ms_ssim')):
            try:
                import piq
            except ImportError as exc:
                raise ImportError('aux.gmsd / fsim_surrogate / ms_ssim need the `piq` package') from exc
            if self.weights['gmsd'] > 0:
                self.gmsd = piq.GMSDLoss(reduction='none', data_range=1.0)
            if self.weights['fsim'] > 0:
                self.fsim = piq.FSIMLoss(reduction='none', data_range=1.0, chromatic=False)
            if self.weights['ms_ssim'] > 0:
                self.ms_ssim = piq.MultiScaleSSIMLoss(reduction='none', data_range=1.0)
        kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]) / 16.0
        self.register_buffer('scharr_x', kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('scharr_y', kx.t().contiguous().view(1, 1, 3, 3), persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        super(StructureLosses, self).train(False)

    def train(self, mode=True):
        return super(StructureLosses, self).train(False)

    @property
    def active(self):
        return [name for name, w in self.weights.items() if w > 0]

    def _scharr(self, gray):
        gx = F.conv2d(gray, self.scharr_x.to(gray.dtype), padding=1)
        gy = F.conv2d(gray, self.scharr_y.to(gray.dtype), padding=1)
        return gx, gy

    def log_radial_spectrum(self, gray, cutoff=TEXTURE_CUTOFF):
        """(n, 1, S, S) grey [0, 1] -> (n, S//2 - cutoff) log10 radially averaged power spectrum for radii
        cutoff..S//2-1 (cycles per image). Mean removed; no window, like the scoring images' own statistics."""
        n, _, h, w = gray.shape
        key = (h, w, str(gray.device))
        if key not in self._radial:
            fy = torch.fft.fftfreq(h, device=gray.device) * h
            fx = torch.fft.fftfreq(w, device=gray.device) * w
            r = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).round().long().clamp(max=min(h, w) // 2 - 1)
            counts = torch.bincount(r.view(-1), minlength=min(h, w) // 2).float()
            self._radial[key] = (r.view(-1), counts)
        r, counts = self._radial[key]
        g = gray[:, 0] - gray[:, 0].mean(dim=(1, 2), keepdim=True)
        power = torch.fft.fft2(g).abs() ** 2 / float(h * w)
        sums = torch.zeros(n, counts.numel(), device=gray.device, dtype=power.dtype)
        sums.index_add_(1, r, power.view(n, -1))
        return torch.log10(sums / counts.clamp(min=1.0) + 1e-10)[:, cutoff:]

    def forward(self, pred, target, timesteps):
        """pred, target (n, 3, S, S) float32 [-1, 1]; timesteps (n,) →
        {name: (per-sample loss (n,), eligible rows (n,) bool)} for every active term."""
        n = pred.shape[0]
        out = {}
        with autocast_disabled(pred.device):
            pred = pred.float()
            target = target.float()
            everyone = torch.ones(n, dtype=torch.bool, device=pred.device)
            if self.lpips is not None:
                out['lpips'] = (self.lpips(pred, target).view(n), everyone)
            gray_needed = any(self.weights[k] > 0 for k in ('scharr', 'gmsd', 'fsim', 'ms_ssim', 'texture'))
            if gray_needed:
                gp, gt = to_gray01(pred), to_gray01(target)
            if self.weights['scharr'] > 0:
                pgx, pgy = self._scharr(gp)
                tgx, tgy = self._scharr(gt)
                pm = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-8)
                tm = torch.sqrt(tgx ** 2 + tgy ** 2 + 1e-8)
                dims = (1, 2, 3)
                value = (pm - tm).abs().mean(dim=dims) + 0.5 * ((pgx - tgx).abs().mean(dim=dims)
                                                               + (pgy - tgy).abs().mean(dim=dims))
                out['scharr'] = (value, everyone)
            if self.gmsd is not None:
                out['gmsd'] = (self.gmsd(gp, gt).view(n), everyone)
            if self.weights['texture'] > 0:
                eligible = timesteps.to(pred.device) <= self.max_t['texture']
                with torch.no_grad():
                    ts = self.log_radial_spectrum(gt)
                value = (self.log_radial_spectrum(gp) - ts).abs().mean(dim=1)
                out['texture'] = (value * eligible.float(), eligible)
            for name, fn in (('fsim', self.fsim), ('ms_ssim', self.ms_ssim)):
                if fn is None:
                    continue
                eligible = timesteps.to(pred.device) <= self.max_t[name]
                value = pred.new_zeros(n)
                if bool(eligible.any()):
                    idx = torch.nonzero(eligible, as_tuple=False).view(-1)
                    value = value.index_put((idx,), fn(gp[idx], gt[idx]).view(-1).float())
                out[name] = (value, eligible)
        return out
