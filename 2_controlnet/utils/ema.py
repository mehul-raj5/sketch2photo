"""EMA of trainable parameters **and buffers** (P9, NEW-pipeline-9; port of the July template's EMA).

- Shadow copies live on the modules' device; floating tensors are averaged in float32, integer buffers copied.
- `update()` after every optimiser step; `state_dict()` goes into the full-state checkpoint (rotated with it).
- `swapped()` puts the averaged weights into the live modules for validation / generation and restores the raw
  ones afterwards; `copy_state_into(modules, state)` loads an EMA checkpoint straight into modules (generation).
"""
import contextlib
from collections import OrderedDict

import torch


def _named_tensors(modules, include_buffers):
    for mname, module in modules.items():
        for name, param in module.named_parameters():
            yield '%s.param.%s' % (mname, name), param
        if include_buffers:
            for name, buf in module.named_buffers():
                yield '%s.buffer.%s' % (mname, name), buf


class EMA(object):
    def __init__(self, modules, decay, include_buffers=True):
        self.modules = OrderedDict(modules)
        self.decay = float(decay)
        self.include_buffers = bool(include_buffers)
        self.num_updates = 0
        self.shadow = OrderedDict()
        for key, tensor in _named_tensors(self.modules, self.include_buffers):
            t = tensor.detach()
            self.shadow[key] = t.clone().float() if t.is_floating_point() else t.clone()
        self._backup = None

    @torch.no_grad()
    def update(self):
        d = self.decay
        for key, tensor in _named_tensors(self.modules, self.include_buffers):
            s = self.shadow[key]
            if tensor.is_floating_point():
                s.mul_(d).add_(tensor.detach().to(device=s.device, dtype=s.dtype), alpha=1.0 - d)
            else:
                s.copy_(tensor.detach())
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self):
        for key, tensor in _named_tensors(self.modules, self.include_buffers):
            tensor.copy_(self.shadow[key].to(device=tensor.device, dtype=tensor.dtype))

    @torch.no_grad()
    def store(self):
        self._backup = OrderedDict((k, t.detach().clone()) for k, t in _named_tensors(self.modules, self.include_buffers))

    @torch.no_grad()
    def restore(self):
        if self._backup is None:
            raise RuntimeError('EMA.restore() called before store()')
        for key, tensor in _named_tensors(self.modules, self.include_buffers):
            tensor.copy_(self._backup[key])
        self._backup = None

    @contextlib.contextmanager
    def swapped(self):
        self.store()
        self.copy_to()
        try:
            yield self
        finally:
            self.restore()

    def state_dict(self):
        return {'decay': self.decay, 'include_buffers': self.include_buffers, 'num_updates': self.num_updates,
                'shadow': OrderedDict((k, v.detach().cpu()) for k, v in self.shadow.items())}

    @torch.no_grad()
    def load_state_dict(self, state):
        _check_keys(set(self.shadow), set(state['shadow']))
        for key, value in state['shadow'].items():
            self.shadow[key].copy_(value.to(device=self.shadow[key].device, dtype=self.shadow[key].dtype))
        self.decay = float(state['decay'])
        self.num_updates = int(state.get('num_updates', 0))

    @staticmethod
    @torch.no_grad()
    def copy_state_into(modules, state):
        """Write an EMA checkpoint's averaged weights into live modules (strict on keys)."""
        modules = OrderedDict(modules)
        named = OrderedDict(_named_tensors(modules, bool(state['include_buffers'])))
        _check_keys(set(named), set(state['shadow']))
        for key, tensor in named.items():
            tensor.copy_(state['shadow'][key].to(device=tensor.device, dtype=tensor.dtype))
        return int(state.get('num_updates', 0))


def _check_keys(mine, theirs):
    if mine != theirs:
        raise RuntimeError('EMA state does not match the modules: missing %s, unexpected %s'
                           % (sorted(mine - theirs)[:5], sorted(theirs - mine)[:5]))
