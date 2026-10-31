"""ours training — step-based and resumable (P3, P5–P11, CN-L4; PLAN P1.2 / P1.4).

New run:  python train_controlnet.py --config 2_controlnet/configs/wildsketch.yaml --device cuda:0 [--run_id NAME] [--set key=value ...]
Resume:   python train_controlnet.py --resume <outputs>/<id>/checkpoints/<run_id> --device cuda:0

- Every schedule counts optimiser steps: LR warm-up + cosine, logging, checkpoints, validation (D7, NEW-pipeline-11).
- AdamW with two learning rates (CN-L4): ControlNet 1e-5; adapter + new conv_in columns + 13 zero convs 5e-5;
  no weight decay on norms and biases; one gradient clip over all trainable parameters.
- EMA of parameters and buffers after every optimiser step (P9).
- Atomic full-state checkpoints every `checkpoint.every_steps`, at the end and on SIGTERM / SIGINT; the last N kept.
  Resume restores weights, EMA, optimiser, scaler, step, data position, every RNG and the step generator; only
  logging / retention / worker / val-cadence keys may change on resume (P3, NEW-pipeline-1).
- Validation every `val.every_steps` and at the end (split `train_dev` only): ε-MSE at fixed t on the val pairs
  and on a train subset with the raw weights (M11), then 256² PNGs of the val sketches with the inference config,
  the EMA weights and per-image seeds into <outputs>/<id>/val/<run_id>/step_<N>/ (P11).
- The resolved config, code sha256 and input sha256 are written at every (re)start and into every checkpoint (rule 7).
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import time

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from utils import config as C  # noqa: E402  (no torch import)
from data.prompts import Prompts  # noqa: E402
from utils.checkpoint import list_checkpoints, resolve_checkpoint  # noqa: E402  (torch imported lazily)

RESUME_OVERRIDABLE = ('data.num_workers', 'data.pin_memory', 'log', 'checkpoint.keep_last', 'checkpoint.keep_every', 'val.every_steps',
                      'val.max_images', 'val.write_png')
# --extend (fork) may additionally lengthen the run; nothing about the method itself.
EXTEND_OVERRIDABLE = ('optim.total_steps', 'optim.schedule_steps', 'checkpoint.every_steps')
# --extend --refine: a refinement fork may also switch on the texture term and restart the LR (2026-09-24 texture study).
REFINE_OVERRIDABLE = ('aux.texture', 'optim.restart')


def parse_args():
    p = argparse.ArgumentParser(description='ours training (step-based, resumable)')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--config', help='YAML config for a new run')
    src.add_argument('--resume', help='a run checkpoint folder (newest checkpoint is used) or one step_XXXXXXX folder')
    src.add_argument('--extend', help='fork a checkpoint into a NEW run (--run_id) and train it further: full state '
                                      'is loaded, the source run is never written, the cosine stays pinned to the '
                                      'source total_steps and the extra steps run at the floor LR (set '
                                      'optim.total_steps higher)')
    p.add_argument('--device', required=True, help='torch device, e.g. cuda:0 — never defaulted (D12)')
    p.add_argument('--run_id', default=None, help='new runs only; overrides run.run_id')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='config override, value parsed as YAML; repeatable')
    p.add_argument('--refine', action='store_true',
                   help='with --extend: also allow aux.texture.* and optim.restart.* overrides (a refinement fork)')
    p.add_argument('--stop_after', type=int, default=None,
                   help='checkpoint and exit after this many optimiser steps in this session (exact-resume test M3)')
    return p.parse_args()


def seed_everything(seed, deterministic):
    import random

    import numpy as np
    import torch
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def lr_factor(step, optim_cfg):
    """Linear warm-up over `warmup_steps`, then cosine from 1 to `min_lr_ratio` at `schedule_steps` (default
    `total_steps`), then constant at `min_lr_ratio` — the fine-tuning tail when schedule_steps < total_steps."""
    warm, floor = optim_cfg['warmup_steps'], optim_cfg['min_lr_ratio']
    restart = optim_cfg.get('restart') or {}
    if restart.get('step') is not None and step >= restart['step']:
        local, r_warm, peak = step - restart['step'], restart['warmup'], restart['peak']
        r_total = optim_cfg['total_steps'] - restart['step']
        if r_warm > 0 and local < r_warm:
            return floor + (peak - floor) * float(local + 1) / float(r_warm)
        progress = min(1.0, float(local - r_warm) / float(max(1, r_total - r_warm)))
        return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    total = optim_cfg.get('schedule_steps') or optim_cfg['total_steps']
    if warm > 0 and step < warm:
        return float(step + 1) / float(warm)
    progress = min(1.0, float(step - warm) / float(max(1, total - warm)))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(pipe, cfg):
    import torch
    from models.conditioning import is_fast_controlnet_param
    o = cfg['optim']
    groups = {}

    def add(group, lr, decay, param):
        key = (group, bool(decay))
        if key not in groups:
            groups[key] = {'params': [], 'lr': lr, 'base_lr': lr, 'weight_decay': o['weight_decay'] if decay else 0.0,
                           'name': group + ('' if decay else '_no_decay')}
        groups[key]['params'].append(param)

    for name, param in pipe.controlnet.named_parameters():
        if not param.requires_grad:
            continue
        fast = is_fast_controlnet_param(name)
        add('adapter_and_new_columns' if fast else 'controlnet',
            o['lr_adapter_and_new_columns'] if fast else o['lr_controlnet'], param.ndim > 1, param)
    for name, param in pipe.adapter.named_parameters():
        add('adapter_and_new_columns', o['lr_adapter_and_new_columns'], param.ndim > 1, param)
    ordered = [groups[k] for k in sorted(groups)]
    optimizer = torch.optim.AdamW(ordered, lr=o['lr_controlnet'], betas=tuple(o['betas']), eps=o['eps'])
    summary = [{'name': g['name'], 'lr': g['base_lr'], 'weight_decay': g['weight_decay'],
                'tensors': len(g['params']), 'parameters': int(sum(p.numel() for p in g['params']))} for g in ordered]
    return optimizer, summary


def make_scaler(enabled, device):
    import torch
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def grad_norm(params):
    import torch
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def stem_seed(*parts):
    return int(hashlib.sha256('/'.join(str(p) for p in parts).encode('utf-8')).hexdigest()[:8], 16) % (2 ** 31)


def eps_mse_at_fixed_t(pipe, dataset, cfg, text_cond):
    """Mean unweighted ε-MSE at each t in val.loss_t with per-stem fixed noise, no dropout, scale = residual_scale."""
    import torch
    import torch.nn.functional as F
    from data.sd_dataset import make_eval_loader
    from utils.precision import autocast_ctx
    loader = make_eval_loader(dataset, cfg['optim']['batch'], num_workers=min(2, cfg['data']['num_workers']),
                              pin_memory=False, seed=0)
    ts = list(cfg['val']['loss_t'])
    sums = dict((t, 0.0) for t in ts)
    count = 0
    precision = cfg['run']['precision']
    scale = cfg['conditioning']['residual_scale']
    with torch.no_grad():
        for batch in loader:
            photo = batch['photo'].to(pipe.device)
            bsz = photo.shape[0]
            z0 = pipe.encode_latents_mean(photo, precision)
            with autocast_ctx(pipe.device, precision):
                c_spatial = pipe.conditioning(batch['sketch'].to(pipe.device), batch['sketch_raw'].to(pipe.device))
            text = text_cond.expand(bsz, -1, -1)
            for t_value in ts:
                noise = torch.stack([torch.randn(tuple(z0.shape[1:]), generator=torch.Generator().manual_seed(
                    stem_seed('eps_mse', stem, t_value))) for stem in batch['stem']]).to(pipe.device)
                t = torch.full((bsz,), int(t_value), dtype=torch.long, device=pipe.device)
                z_t = pipe.noise_scheduler.add_noise(z0, noise, t)
                with autocast_ctx(pipe.device, precision):
                    down, mid = pipe.controlnet_residuals(z_t, t, text, c_spatial, scale)
                    eps = pipe.unet_eps(z_t, t, text, down, mid)
                sums[t_value] += float(F.mse_loss(eps.float(), noise, reduction='none').mean(dim=(1, 2, 3)).sum())
            count += bsz
    return dict((str(t), sums[t] / max(1, count)) for t in ts)


def main():
    args = parse_args()
    resume_dir = None
    if args.resume:
        if args.run_id:
            raise SystemExit('--run_id cannot be combined with --resume (the run id comes from the checkpoint)')
        resume_dir = resolve_checkpoint(args.resume)
        cfg = C.load_config(os.path.join(resume_dir, 'config_resolved.yaml'), args.overrides,
                            allowed_prefixes=RESUME_OVERRIDABLE)
    elif args.extend:
        # Fork: resume the full state of another run's checkpoint, but write to a new run id, so the source run's
        # checkpoints are never pruned or overwritten. Only the length (and bookkeeping) may change: the method
        # itself stays the source's.
        if not args.run_id:
            raise SystemExit('--extend needs --run_id for the new run')
        resume_dir = resolve_checkpoint(args.extend)
        src_cfg_path = os.path.join(resume_dir, 'config_resolved.yaml')
        source = C.load_config(src_cfg_path)
        cfg = C.load_config(src_cfg_path, args.overrides,
                            allowed_prefixes=RESUME_OVERRIDABLE + EXTEND_OVERRIDABLE
                            + (REFINE_OVERRIDABLE if args.refine else ()))
        if cfg['optim']['schedule_steps'] is None:
            cfg['optim']['schedule_steps'] = source['optim']['schedule_steps'] or source['optim']['total_steps']
        if args.run_id == source['run']['run_id']:
            raise SystemExit('--extend must write to a different run id than its source (%s)' % source['run']['run_id'])
        cfg['run']['run_id'] = args.run_id
        C.validate(C.plain(cfg))
    else:
        if args.refine:
            raise SystemExit('--refine only works together with --extend')
        cfg = C.load_config(args.config, args.overrides)
        if args.run_id:
            cfg['run']['run_id'] = args.run_id
            C.validate(C.plain(cfg))
        if cfg['run']['run_id'] is None:
            cfg['run']['run_id'] = C.default_run_id(cfg)
    C.apply_env(cfg)  # HF_HOME / TORCH_HOME before diffusers / transformers are imported

    import torch
    import yaml
    from data.sd_dataset import (EpochBatchSampler, PairedSquashDataset, load_landmarks,
                                 make_continuous_train_loader, training_stems)
    from data.sketch_dataset import SketchOnlyDataset
    from models.conditioning import column_norms
    from models.controlnet_pipeline import SketchToPhotoPipeline
    from models.face_identity_loss import AlignedIdentityLoss
    from models.structure_losses import StructureLosses
    from utils.checkpoint import capture_rng, load_part, restore_rng, save_checkpoint
    from utils.ema import EMA
    from utils.logging_utils import MeanMeter, RunLogger, TimestepBins, append_run_record, environment_record

    run_id = cfg['run']['run_id']
    r = cfg['_resolved']
    dirs = C.run_dirs(cfg, run_id)
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    if (resume_dir is None or args.extend) and list_checkpoints(dirs['checkpoints']):
        raise SystemExit('run %s already has checkpoints in %s — use --resume on it, or another --run_id'
                         % (run_id, dirs['checkpoints']))
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    seed = cfg['run']['seed']
    seed_everything(seed, cfg['run']['deterministic'])
    trainer_state = load_part(resume_dir, 'trainer_state') if resume_dir else None
    start_step = int(trainer_state['step']) if trainer_state else 0
    if args.extend and cfg['optim']['total_steps'] <= start_step:
        raise SystemExit('--extend: optim.total_steps (%d) must exceed the source step (%d)'
                         % (cfg['optim']['total_steps'], start_step))
    logger = RunLogger(dirs['logs'], cfg['log']['tensorboard'], start_step=start_step)
    logger.info('run %s — %s at step %d on %s' % (run_id, 'resume from %s' % resume_dir if resume_dir else 'start',
                                                  start_step, device))

    # ---------------------------------------------------------------- data
    o = cfg['optim']
    photo_size, sketch_size = cfg['data']['photo_size'], cfg['data']['sketch_size']
    train_stems = training_stems(r['dataset_dir'], r['splits_dir'], cfg['run']['split'])
    id_on = cfg['aux']['id']['weight'] > 0
    landmarks = None
    if id_on:
        landmarks = load_landmarks(r['landmarks'])
        if landmarks['dataset'] != cfg['run']['dataset']:
            raise SystemExit('landmark file is for %s, run is %s' % (landmarks['dataset'], cfg['run']['dataset']))
    train_prompts = Prompts(cfg, cfg['run']['split'], r)
    logger.info('prompts: %s' % train_prompts.describe())
    train_ds = PairedSquashDataset(r['dataset_dir'], train_stems, photo_size, sketch_size, flip=cfg['data']['flip'],
                                   seed=seed, landmarks=landmarks, prompts=train_prompts)
    bpe = EpochBatchSampler.batches_per_epoch(len(train_ds), o['batch'])
    if bpe < 1:
        raise SystemExit('batch %d is larger than the %d training pairs' % (o['batch'], len(train_ds)))
    val_enabled = cfg['run']['split'] != 'all'
    val_pairs = train_probe = val_sketches = None
    if val_enabled:
        val_pairs = PairedSquashDataset(r['dataset_dir'], training_stems(r['dataset_dir'], r['splits_dir'], 'val'),
                                        photo_size, sketch_size, flip=0.0, seed=seed)
        n_probe = min(cfg['val']['loss_train_stems'], len(train_stems))
        if n_probe > 0:
            train_probe = PairedSquashDataset(r['dataset_dir'], train_stems[:n_probe], photo_size, sketch_size,
                                              flip=0.0, seed=seed)
        val_sketches = SketchOnlyDataset(r['dataset_dir'], r['splits_dir'], 'val', photo_size, sketch_size,
                                         limit=cfg['val']['max_images'], prompts=Prompts(cfg, 'val', r))
    else:
        logger.info('run.split = all: validation disabled (val stems are training pairs, D5)')

    # ---------------------------------------------------------------- model, losses, optimiser, EMA
    # The val hook generates with the run's own inference block; an x̂ start needs the frozen decoder.
    if cfg['anchor']['enabled'] and cfg['inference']['start'] != 'xhat':
        raise ValueError('anchor.enabled trains for the x̂-anchored regime and needs inference.start: xhat')
    # The frozen decoder produces x̂: needed by the val hook when inference starts from x̂, and by anchored training.
    pipe = SketchToPhotoPipeline(cfg, device, with_decoder=(cfg['anchor']['enabled'] or
                                                            (val_enabled and cfg['inference']['start'] == 'xhat')))
    structure = StructureLosses(cfg['aux'])
    id_loss = AlignedIdentityLoss(r['ir_se50'], cfg['aux']['id']['weights_sha256']) if id_on else None
    pipe.attach_losses(structure if structure.active else None, id_loss)
    optimizer, group_summary = build_optimizer(pipe, cfg)
    use_scaler = cfg['run']['precision'] == 'fp16'
    scaler = make_scaler(use_scaler, device)
    ema = EMA(pipe.trainable_modules(), cfg['ema']['decay'], cfg['ema']['include_buffers'])
    step_generator = torch.Generator()
    step_generator.manual_seed(seed)
    text_cond = pipe.encode_prompt([cfg['prompt']['text']])
    text_uncond = pipe.encode_prompt([''])

    step, epoch, batch_in_epoch = 0, 0, 0
    if resume_dir:
        trainable = load_part(resume_dir, 'trainable')
        pipe.controlnet.load_state_dict(trainable['controlnet'], strict=True)
        pipe.adapter.load_state_dict(trainable['adapter'], strict=True)
        del trainable
        ema.load_state_dict(load_part(resume_dir, 'ema'))
        opt_state = load_part(resume_dir, 'optimizer')
        optimizer.load_state_dict(opt_state['optimizer'])
        if use_scaler:
            scaler.load_state_dict(opt_state['scaler'])
        del opt_state
        if int(trainer_state['batches_per_epoch']) != bpe:
            raise SystemExit('batches per epoch changed (%s -> %d): dataset or batch differ from the checkpoint'
                             % (trainer_state['batches_per_epoch'], bpe))
        step = int(trainer_state['step'])
        epoch = int(trainer_state['epoch'])
        batch_in_epoch = int(trainer_state['batch_in_epoch'])
        step_generator.set_state(trainer_state['generator'])
        restore_rng(trainer_state['rng'], device)  # last: nothing below consumes RNG before the first step

    code_hashes = C.code_sha256()
    inputs = {'codebook_file': pipe.cvq.file_sha256, 'codebook_embedding': pipe.cvq.embedding_sha256}
    for name in ('train_dev', 'val'):
        split_file = os.path.join(r['splits_dir'], name + '.txt')
        if os.path.isfile(split_file):
            inputs['split_' + name] = C.sha256_file(split_file)
    if id_on:
        inputs['landmarks'] = C.sha256_file(r['landmarks'])
        inputs['ir_se50'] = id_loss.weights_sha256
    env = environment_record(device)
    C.dump_config(cfg, os.path.join(dirs['logs'], 'config_resolved.yaml'))
    append_run_record(dirs['logs'], {
        'event': 'resume' if resume_dir else 'start', 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'step': step,
        'argv': sys.argv, 'resume_from': resume_dir, 'overrides': r.get('overrides'), 'environment': env,
        'code_sha256': code_hashes, 'inputs_sha256': inputs, 'pipeline': pipe.info, 'optimizer_groups': group_summary,
        'train_pairs': len(train_ds), 'landmarks_found': train_ds.landmarks_found, 'batches_per_epoch': bpe,
        'loss_terms': structure.active + (['id'] if id_on else []) +
                      (['latent_l1_scharr'] if cfg['aux']['latent_l1_scharr'] > 0 else [])})
    logger.info('pairs %d · batches/epoch %d · loss terms %s · groups %s'
                % (len(train_ds), bpe, structure.active + (['id'] if id_on else []),
                   ['%s lr %.1e n %d' % (g['name'], g['lr'], g['parameters']) for g in group_summary]))

    # ---------------------------------------------------------------- checkpoint and validation helpers
    def save(step_now, reason):
        integrity = pipe.cvq.assert_integrity()
        state = {'step': step_now, 'epoch': epoch, 'batch_in_epoch': batch_in_epoch, 'batches_per_epoch': bpe,
                 'generator': step_generator.get_state(), 'rng': capture_rng(device), 'run_id': run_id,
                 'reason': reason, 'time': time.time()}
        payload = {'trainable': {'controlnet': pipe.controlnet.state_dict(), 'adapter': pipe.adapter.state_dict()},
                   'ema': ema.state_dict(),
                   'optimizer': {'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if use_scaler else None},
                   'trainer_state': state}
        meta = {'run_id': run_id, 'step': step_now, 'reason': reason, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'code_sha256': code_hashes, 'inputs_sha256': inputs, 'codebook_integrity': integrity,
                'ema_updates': ema.num_updates, 'environment': env}
        texts = {'config_resolved.yaml': yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None),
                 'meta.json': json.dumps(meta, indent=1, sort_keys=True, default=str)}
        path, seconds, removed = save_checkpoint(dirs['checkpoints'], step_now, payload, texts,
                                                 cfg['checkpoint']['keep_last'], cfg['checkpoint'].get('keep_every'))
        logger.info('checkpoint %s (%s, %.1f s, %d old removed)' % (path, reason, seconds, len(removed)))
        logger.metrics(step_now, 'checkpoint', {'seconds': seconds, 'reason': reason})

    def val_dir(step_now, seed_value):
        name = ('step_%d' % step_now) if len(cfg['val']['seeds']) == 1 else ('step_%d_seed%d' % (step_now, seed_value))
        return os.path.join(dirs['val'], name)

    def run_validation(step_now):
        from utils.generation import run_generation
        t0 = time.time()
        pipe.eval_mode()
        record = {'eps_mse_val': eps_mse_at_fixed_t(pipe, val_pairs, cfg, text_cond)}
        if train_probe is not None:
            record['eps_mse_train'] = eps_mse_at_fixed_t(pipe, train_probe, cfg, text_cond)
        if cfg['val']['write_png']:
            swap = ema.swapped() if cfg['ema']['eval_with_ema'] else contextlib.nullcontext()
            with swap:
                for seed_value in cfg['val']['seeds']:
                    out = run_generation(pipe, cfg, val_sketches, val_dir(step_now, seed_value), seed_value,
                                         extra={'source': 'train_controlnet.py val hook', 'run_id': run_id,
                                                'step': step_now, 'weights': 'ema' if cfg['ema']['eval_with_ema'] else 'raw',
                                                'code_sha256': code_hashes, 'inputs_sha256': inputs},
                                         overwrite=True, log=logger.info)
                    record['png_dir_seed%d' % seed_value] = out['out_dir']
        pipe.train_mode()
        record['seconds'] = time.time() - t0
        logger.metrics(step_now, 'val', record)
        logger.info('val @ %d: eps-MSE val %s train %s (%.0f s)' % (step_now, record['eps_mse_val'],
                                                                   record.get('eps_mse_train'), record['seconds']))

    stop = {'requested': False}

    def on_signal(signum, frame):
        if stop['requested']:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        stop['requested'] = True
        logger.info('signal %d: checkpoint after the current step, then exit (send again to abort)' % signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    total_steps = o['total_steps']
    if resume_dir and val_enabled and cfg['val']['write_png'] and step > 0 and \
            (step % cfg['val']['every_steps'] == 0 or step == total_steps) and \
            not os.path.isfile(os.path.join(val_dir(step, cfg['val']['seeds'][0]), 'run.json')):
        logger.info('val outputs of step %d incomplete: re-running validation' % step)
        run_validation(step)

    # ---------------------------------------------------------------- loop
    trainable_params = [p for g in optimizer.param_groups for p in g['params']]
    adapter_params = list(pipe.adapter.parameters())
    meter, bins = MeanMeter(), TimestepBins()
    pipe.train_mode()
    loader = make_continuous_train_loader(train_ds, o['batch'], seed, epoch, batch_in_epoch,
                                          cfg['data']['num_workers'], cfg['data']['pin_memory'])
    batches = iter(loader)
    session_steps, since_log, nonfinite_streak = 0, 0, 0
    last_saved = step if resume_dir else -1
    t_log = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    while step < total_steps:
        try:
            batch = next(batches)
        except Exception:
            if stop['requested']:
                if last_saved != step:
                    save(step, 'signal')
                logger.close()
                return 0
            raise
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * lr_factor(step, o)
        cond = pipe.encode_prompt(list(batch['prompt'])) if train_prompts.enabled else text_cond
        # v1 two-phase schedule (aux.boost): a pure function of the step, so a resumed run gets the same weights.
        boost = cfg['aux']['boost']
        boosted = 0 < boost['start_step'] <= step
        pipe.aux_multipliers = {'lpips': boost['lpips'] if boosted else 1.0, 'id': boost['id'] if boosted else 1.0}
        loss, scalars, per_sample, per_row = pipe.training_step(batch, cond, text_uncond, step_generator)
        batch_in_epoch += 1
        if batch_in_epoch >= bpe:
            epoch, batch_in_epoch = epoch + 1, 0
        if not torch.isfinite(loss):
            nonfinite_streak += 1
            optimizer.zero_grad(set_to_none=True)
            logger.info('non-finite loss at step %d (streak %d): batch skipped' % (step, nonfinite_streak))
            if nonfinite_streak >= 5:
                raise RuntimeError('5 consecutive non-finite losses at step %d' % step)
            continue
        nonfinite_streak = 0
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        adapter_norm = grad_norm(adapter_params)
        total_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, o['grad_clip']))
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update()
        step += 1
        session_steps += 1
        since_log += 1

        lrs = dict((g['name'], g['lr']) for g in optimizer.param_groups)
        scalars.update({'grad_norm': total_norm, 'grad_norm_adapter': adapter_norm,
                        'adapter_grad_share': adapter_norm / max(total_norm, 1e-12),
                        'lr_controlnet': lrs.get('controlnet', 0.0),
                        'lr_adapter_and_new_columns': lrs.get('adapter_and_new_columns', 0.0)})
        if use_scaler:
            scalars['grad_scale'] = float(scaler.get_scale())
        meter.add(scalars)
        bins.add('mse', per_sample['t'], per_sample['mse'])
        if 'latent' in per_sample:
            bins.add('latent', per_sample['t'], per_sample['latent'])
        for name, (t_rows, values) in per_row.items():
            bins.add(name, t_rows, values)

        if step == 1 or step % cfg['log']['every_steps'] == 0 or step == total_steps:
            summary = meter.summary()
            summary.update(column_norms(pipe.controlnet))
            summary.update({'sec_per_step': (time.time() - t_log) / max(1, since_log), 'epoch': epoch,
                            'bins': bins.summary()})
            if device.type == 'cuda':
                summary['max_mem_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            logger.metrics(step, 'train', summary)
            logger.info('step %d/%d  loss %.4f  mse %.4f  aux %d/%d  lr %.2e  %.2f s/step  %.1f GB'
                        % (step, total_steps, summary.get('loss_total', float('nan')), summary.get('loss_mse', float('nan')),
                           round(summary.get('aux_decoded', 0)), round(summary.get('aux_samples', 0)),
                           summary.get('lr_controlnet', 0.0), summary['sec_per_step'], summary.get('max_mem_gb', 0.0)))
            meter.reset()
            bins.reset()
            t_log, since_log = time.time(), 0

        stop_session = args.stop_after is not None and session_steps >= args.stop_after
        if (step % cfg['checkpoint']['every_steps'] == 0 or step == total_steps or stop['requested'] or stop_session) \
                and last_saved != step:
            save(step, 'end' if step == total_steps else ('signal' if stop['requested'] else
                                                         ('stop_after' if stop_session else 'cadence')))
            last_saved = step
        if val_enabled and (step % cfg['val']['every_steps'] == 0 or step == total_steps) and not stop['requested']:
            run_validation(step)
        if stop['requested'] or stop_session:
            logger.info('stopping at step %d (%s)' % (step, 'signal' if stop['requested'] else '--stop_after'))
            logger.close()
            return 0

    logger.info('training finished at step %d' % step)
    logger.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
