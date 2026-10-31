"""Dual-path ControlNet conditioning: 3 soft-edge channels + N adapter channels (CN-L1, CN-L2, CN-L4, CN-L5, CN-L6).

- `DualPathConvIn` replaces `controlnet_cond_embedding.conv_in` of the scribble ControlNet. The pretrained
  columns (0-2, edge path) and the new adapter columns (3..) are separate parameters, so the optimiser can
  give the new columns their own learning-rate group (CN-L4). The new columns start at zero in every code
  path, so step 0 is exactly the pretrained scribble ControlNet on the edge path (NEW-controlnet-2).
  Its state_dict exposes one merged `weight` (out, 3+N, k, k) and `bias` — the stock diffusers layout —
  and `register_to_config(conditioning_channels=3+N)` keeps config.json consistent with it (NEW-controlnet-3).
- `soft_edges`: v1's tone map (luma → 3×3 mean → 1 − x → × gain → clamp), one gain for training and
  inference (NEW-controlnet-5); with `border_fix` the mean ignores the zero padding, so the 1-px frame of
  0.5 / 0.83 disappears (MC16, NEW-pipeline-8).
- `draw_condition_dropout` / `assemble_conditioning`: per-sample exclusive events — zero everything, zero the
  adapter channels, or zero the edge channels (CN-L5) — and the M3 path modes both / edges_only / adapter_only.
- `draw_residual_scale` / `scale_residuals`: one residual scale for training and inference (CN-L1, F15), with a
  per-step jitter in training only (CN-L5).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EDGE_CHANNELS = 3
DROP_KEEP, DROP_ALL, DROP_ADAPTER, DROP_EDGES = 0, 1, 2, 3
PATH_MODES = ('both', 'edges_only', 'adapter_only')
NEW_COLUMNS_PARAM = 'controlnet_cond_embedding.conv_in.weight_new'


class DualPathConvIn(nn.Module):
    def __init__(self, conv, new_channels):
        super(DualPathConvIn, self).__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError('DualPathConvIn wraps an nn.Conv2d, got %s' % type(conv).__name__)
        if conv.groups != 1 or conv.bias is None or conv.padding_mode != 'zeros':
            raise ValueError('conv_in must be a biased, ungrouped, zero-padded Conv2d')
        self.old_channels = conv.in_channels
        self.new_channels = int(new_channels)
        self.in_channels = self.old_channels + self.new_channels
        self.out_channels = conv.out_channels
        self.kernel_size = tuple(conv.kernel_size)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight_old = nn.Parameter(conv.weight.detach().clone())
        self.weight_new = nn.Parameter(torch.zeros((self.out_channels, self.new_channels) + self.kernel_size,
                                                   dtype=conv.weight.dtype))
        self.bias = nn.Parameter(conv.bias.detach().clone())

    @property
    def weight(self):
        return torch.cat([self.weight_old, self.weight_new], dim=1)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    # Stock layout in every state_dict (checkpoints, save_pretrained): one `weight`, one `bias`.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight = self.weight
        destination[prefix + 'weight'] = weight if keep_vars else weight.detach()
        destination[prefix + 'bias'] = self.bias if keep_vars else self.bias.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        wkey, bkey = prefix + 'weight', prefix + 'bias'
        expected_w = (self.out_channels, self.in_channels) + self.kernel_size
        if wkey not in state_dict:
            missing_keys.append(wkey)
        else:
            w = state_dict[wkey]
            if tuple(w.shape) != expected_w:
                error_msgs.append('size mismatch for %s: checkpoint %s, model %s' % (wkey, tuple(w.shape), expected_w))
            else:
                with torch.no_grad():
                    self.weight_old.copy_(w[:, :self.old_channels])
                    self.weight_new.copy_(w[:, self.old_channels:])
        if bkey not in state_dict:
            missing_keys.append(bkey)
        else:
            b = state_dict[bkey]
            if tuple(b.shape) != (self.out_channels,):
                error_msgs.append('size mismatch for %s: checkpoint %s, model %s'
                                  % (bkey, tuple(b.shape), (self.out_channels,)))
            else:
                with torch.no_grad():
                    self.bias.copy_(b)
        if strict:
            for key in state_dict:
                if key.startswith(prefix) and key[len(prefix):].split('.', 1)[0] not in ('weight', 'bias'):
                    unexpected_keys.append(key)


def build_controlnet(controlnet_id, adapter_channels):
    """Scribble ControlNet with conv_in widened to 3 + adapter_channels inputs (new columns zero), config registered."""
    from diffusers import ControlNetModel
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    embedding = controlnet.controlnet_cond_embedding
    old = embedding.conv_in
    if isinstance(old, DualPathConvIn):
        raise RuntimeError('conv_in is already widened')
    if old.in_channels != EDGE_CHANNELS:
        raise ValueError('%s: conv_in has %d input channels, expected %d (scribble init, F29)'
                         % (controlnet_id, old.in_channels, EDGE_CHANNELS))
    embedding.conv_in = DualPathConvIn(old, adapter_channels)
    controlnet.register_to_config(conditioning_channels=EDGE_CHANNELS + int(adapter_channels))
    return controlnet


def is_fast_controlnet_param(name):
    """CN-L4 group A inside the ControlNet: the new conv_in columns and the 13 zero convolutions."""
    return (name == NEW_COLUMNS_PARAM or name.startswith('controlnet_down_blocks.')
            or name.startswith('controlnet_mid_block.'))


def column_norms(controlnet):
    """M4 logging: norms of the two conv_in paths and of the zero convolutions."""
    conv = controlnet.controlnet_cond_embedding.conv_in
    down = torch.stack([m.weight.detach().float().norm() for m in controlnet.controlnet_down_blocks])
    return {'conv_in_edge_norm': float(conv.weight_old.detach().float().norm()),
            'conv_in_adapter_norm': float(conv.weight_new.detach().float().norm()),
            'zero_conv_down_norm': float(down.norm()),
            'zero_conv_mid_norm': float(controlnet.controlnet_mid_block.weight.detach().float().norm())}


def soft_edges(sketch_raw, amplification, border_fix):
    """(B, 3, H, W) sketch in [-1, 1] → (B, 3, H, W) float32 soft edges in [0, 1]."""
    x = (sketch_raw.float() + 1.0) / 2.0
    gray = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
    smooth = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1, count_include_pad=not border_fix)
    edges = ((1.0 - smooth) * float(amplification)).clamp(0.0, 1.0)
    return edges.repeat(1, EDGE_CHANNELS, 1, 1)


def draw_condition_dropout(batch_size, p_all, p_adapter, p_edges, generator):
    """(B,) int64 on CPU: 0 keep · 1 zero all · 2 zero adapter channels · 3 zero edge channels (exclusive)."""
    u = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    a = float(p_all)
    b = a + float(p_adapter)
    c = b + float(p_edges)
    mode = torch.full((batch_size,), DROP_KEEP, dtype=torch.long)
    mode[u < a] = DROP_ALL
    mode[(u >= a) & (u < b)] = DROP_ADAPTER
    mode[(u >= b) & (u < c)] = DROP_EDGES
    return mode


def assemble_conditioning(edges, adapter_out, drop_mode=None, paths='both'):
    """edges (B, 3, H, W) and adapter_out (B, N, H, W) → c_spatial (B, 3+N, H, W) float32."""
    if paths not in PATH_MODES:
        raise ValueError('paths must be one of %s' % (PATH_MODES,))
    edges = edges.float()
    adapter_out = adapter_out.float()
    batch = edges.shape[0]
    keep_e = torch.ones(batch, device=edges.device, dtype=torch.float32)
    keep_a = torch.ones(batch, device=edges.device, dtype=torch.float32)
    if paths == 'edges_only':
        keep_a = torch.zeros_like(keep_a)
    elif paths == 'adapter_only':
        keep_e = torch.zeros_like(keep_e)
    if drop_mode is not None:
        dm = drop_mode.to(edges.device)
        keep_e = keep_e * ((dm != DROP_ALL) & (dm != DROP_EDGES)).float()
        keep_a = keep_a * ((dm != DROP_ALL) & (dm != DROP_ADAPTER)).float()
    return torch.cat([edges * keep_e.view(batch, 1, 1, 1), adapter_out * keep_a.view(batch, 1, 1, 1)], dim=1)


def draw_residual_scale(base, jitter, generator):
    """Training: base × U[lo, hi] once per step (CN-L5); `jitter` None → base."""
    if not jitter:
        return float(base)
    lo, hi = float(jitter[0]), float(jitter[1])
    u = float(torch.rand(1, generator=generator, dtype=torch.float64).item())
    return float(base) * (lo + (hi - lo) * u)


def scale_residuals(down, mid, scale):
    if scale == 1.0:
        return list(down), mid
    return [d * scale for d in down], mid * scale
