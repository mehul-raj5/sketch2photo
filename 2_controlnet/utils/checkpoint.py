"""Atomic, rotated, full-state checkpoints (P3, rule 9; NEW-pipeline-1, NEW-pipeline-6).

<checkpoints>/<run_id>/step_0000500/
  trainable.pt         ControlNet + adapter state_dicts (stock diffusers key layout, 19-channel conv_in)
  ema.pt               EMA shadow of parameters and buffers
  optimizer.pt         AdamW state (+ GradScaler state for fp16)
  trainer_state.pt     step, epoch, batch position, every RNG state, the step generator, reason
  config_resolved.yaml the run's resolved config
  meta.json            step, time, code / input sha256, codebook integrity, environment
Each checkpoint is written into `step_XXXXXXX.tmp`, fsynced and renamed; then only the newest `keep_last`
complete checkpoints are kept and stale `.tmp` / `.old` folders are removed. torch is imported lazily so the
checkpoint of a run can be located before HF_HOME is set.
"""
import os
import re
import shutil
import time

STEP_DIR = re.compile(r'^step_(\d{7})$')
TORCH_PARTS = ('trainable', 'ema', 'optimizer', 'trainer_state')
TEXT_PARTS = ('config_resolved.yaml', 'meta.json')
REQUIRED = tuple(p + '.pt' for p in TORCH_PARTS) + TEXT_PARTS


def step_dir_name(step):
    return 'step_%07d' % int(step)


def capture_rng(device):
    import random

    import numpy as np
    import torch
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    import random

    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        if state.get('cuda') is None:
            raise RuntimeError('checkpoint has no CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda'], device)


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def list_checkpoints(root):
    """[(step, path)] of complete checkpoints, oldest first."""
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        m = STEP_DIR.match(name)
        full = os.path.join(root, name)
        if m and os.path.isdir(full) and all(os.path.isfile(os.path.join(full, f)) for f in REQUIRED):
            out.append((int(m.group(1)), full))
    return sorted(out)


def resolve_checkpoint(path, inference=False):
    """A step_XXXXXXX folder → itself (must be complete); a run folder → its newest complete checkpoint.
    inference=True (generate.py): a step folder without optimizer.pt is accepted — generation never reads the optimizer state
    (checkpoints copied back from the local-disk mirror, 2026-09-27, were copied without it)."""
    path = os.path.abspath(path)
    if STEP_DIR.match(os.path.basename(path)):
        need = [f for f in REQUIRED if not (inference and f == 'optimizer.pt')]
        missing = [f for f in need if not os.path.isfile(os.path.join(path, f))]
        if missing:
            raise FileNotFoundError('incomplete checkpoint %s (missing %s)' % (path, missing))
        return path
    ckpts = list_checkpoints(path)
    if not ckpts:
        raise FileNotFoundError('no complete checkpoint under %s' % path)
    return ckpts[-1][1]


def rotate(root, keep_last, keep_every=None):
    """Keep the newest `keep_last` checkpoints plus every milestone step (step % keep_every == 0), which never counts
    towards keep_last and is never removed (F58: a later fine-tune can start from the plateau, not only the end)."""
    removed = []
    ckpts = [c for c in list_checkpoints(root) if not (keep_every and c[0] % keep_every == 0)]
    if keep_last > 0 and len(ckpts) > keep_last:
        for _, path in ckpts[:-keep_last]:
            shutil.rmtree(path)
            removed.append(path)
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and (name.endswith('.tmp') or name.endswith('.old')):
            shutil.rmtree(full)
            removed.append(full)
    return removed


def save_checkpoint(root, step, payload, texts, keep_last, keep_every=None):
    """payload: {trainable, ema, optimizer, trainer_state} objects; texts: {config_resolved.yaml, meta.json} strings.
    Returns (path, seconds, removed paths)."""
    import torch
    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, step_dir_name(step))
    tmp = final + '.tmp'
    t0 = time.time()
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for part in TORCH_PARTS:
        path = os.path.join(tmp, part + '.pt')
        torch.save(payload[part], path)
        with open(path, 'rb') as fh:
            os.fsync(fh.fileno())
    for name in TEXT_PARTS:
        with open(os.path.join(tmp, name), 'w') as fh:
            fh.write(texts[name])
            fh.flush()
            os.fsync(fh.fileno())
    _fsync_dir(tmp)
    if os.path.exists(final):  # re-save of the same step: swap, never leave the step without a complete copy on disk
        old = final + '.old'
        if os.path.exists(old):
            shutil.rmtree(old)
        os.rename(final, old)
        os.rename(tmp, final)
        shutil.rmtree(old)
    else:
        os.rename(tmp, final)
    _fsync_dir(root)
    removed = rotate(root, keep_last, keep_every)
    return final, time.time() - t0, removed


def load_part(ckpt_dir, part):
    """Load one part of our own checkpoint (trusted: it holds Python / NumPy RNG states, so weights_only=False)."""
    import torch
    path = os.path.join(ckpt_dir, part + '.pt')
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location='cpu')
