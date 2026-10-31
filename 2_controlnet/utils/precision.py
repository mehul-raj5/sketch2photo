"""Autocast helpers. Precision always comes from the config (bf16 | fp16 | fp32), never from the machine (D12)."""
import contextlib

import torch

DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}


def device_type(device):
    return device.type if isinstance(device, torch.device) else str(device).split(':')[0]


def autocast_ctx(device, precision):
    """Autocast to the configured dtype; fp32 → no autocast."""
    if precision not in DTYPES:
        raise ValueError('precision must be one of %s, got %r' % (sorted(DTYPES), precision))
    if precision == 'fp32':
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type(device), dtype=DTYPES[precision])


def autocast_disabled(device):
    """Autocast switched off (for fp32-only computations inside an autocast region)."""
    try:
        return torch.autocast(device_type=device_type(device), enabled=False)
    except (RuntimeError, ValueError):
        return contextlib.nullcontext()
