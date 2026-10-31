"""v2 configuration: schema-validated YAML with paths resolved to data/ (P3, NEW-pipeline-6, CB-L1).

- Every key of the YAML is declared in SCHEMA; unknown, missing or ill-typed keys raise.
- `<id>_cvq` is derived from `run.dataset`, never configured (NEW-codebook-4).
- Relative paths resolve from the repository root (the folder holding data/, 2_controlnet/ and 3_inference/).
  Environment variables SKETCH_PROJECT_ROOT, SKETCH_DATA_ROOT and SKETCH_OUTPUTS_ROOT override.
- HF_HOME and TORCH_HOME default to data/pretrained/{hf_cache,torch_cache}; `apply_env` must run
  before diffusers or transformers are imported.
This module never imports torch, so it is safe for every script and tool.
"""
import copy
import hashlib
import json
import os
import re
import time

import yaml

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2_controlnet
DEFAULT_PROJECT_ROOT = os.path.dirname(CODE_ROOT)                        # repository root

DATASETS = ('wildsketch', 'cuhk', 'fs2k', 'color_feret', 'cufsf')
# D15 / MC3: writer geometry per dataset (W, H)
OUTPUT_SIZE = {'wildsketch': (256, 256), 'fs2k': (256, 256),
               'cuhk': (200, 250), 'color_feret': (200, 250), 'cufsf': (200, 250)}
CONFIG_SOURCE_KEY = '_resolved'


class ConfigError(ValueError):
    pass


class Key(object):
    """A leaf of the schema: kind in {str, int, float, bool, list_int, list_float}."""

    def __init__(self, kind, choices=None, nullable=False, check=None, hint=''):
        self.kind = kind
        self.choices = choices
        self.nullable = nullable
        self.check = check
        self.hint = hint


def _prob(v):
    return 0.0 <= v <= 1.0


def _pos(v):
    return v > 0


def _nonneg(v):
    return v >= 0


def _hex64(v):
    return re.match(r'^[0-9a-f]{64}$', v) is not None


def _run_id(v):
    return re.match(r'^[A-Za-z0-9._-]+$', v) is not None


def _pair_range(v):
    return len(v) == 2 and 0 < v[0] <= v[1]


def _pair_pos(v):
    return len(v) == 2 and all(x > 0 for x in v)


def _betas(v):
    return len(v) == 2 and all(0.0 <= b < 1.0 for b in v)


def _t_list(v):
    return len(v) > 0 and all(0 <= x < 1000 for x in v)


def _flag_block():
    return {'enabled': Key('bool'), 'max_t': Key('int', check=_nonneg), 'weight': Key('float', check=_nonneg)}


SCHEMA = {
    'run': {
        'dataset': Key('str', DATASETS),
        'split': Key('str', ('train_dev', 'all')),
        'seed': Key('int', check=_nonneg),
        'precision': Key('str', ('bf16', 'fp16', 'fp32')),
        'deterministic': Key('bool'),
        'run_id': Key('str', nullable=True, check=_run_id, hint='letters, digits, . _ -'),
    },
    'paths': {
        'data_root': Key('str'),
        'outputs_root': Key('str'),
        'hf_home': Key('str'),
        'torch_home': Key('str'),
        'ir_se50': Key('str'),
        'landmarks': Key('str'),
    },
    # Tone normalisation of the x̂ anchor towards the TRAINING photos (phase 0; F47). Dataset-level prior from
    # the train split — not DECP's per-image AdaIN reference (F12).
    'tone': {
        'enabled': Key('bool'),
        'dir': Key('str'),
        'stats_split': Key('str', ('train_dev', 'all')),
        'method': Key('str', ('lab_moments', 'rgb_moments', 'hist_l')),
        'strength': Key('float', check=_prob),
        'target': Key('str', ('photo',)),
    },
    # Anchored training (plan phase 3, F52): train in the regime we infer in. For a fraction of rows with
    # t_min <= t <= t_max the noisy input is built from the tone-normalised codebook estimate x̂ (as at inference)
    # while the target still points at the real photo: eps* = eps + sqrt(a/(1-a)) (z_xhat - z0).
    'anchor': {
        'enabled': Key('bool'),
        'prob': Key('float', check=lambda v: 0.0 < v <= 1.0, hint='(0, 1]'),
        't_min': Key('int', check=_nonneg),
        't_max': Key('int', check=_nonneg),
    },
    'model': {
        'sd_model_id': Key('str'),
        'vae_id': Key('str'),
        'controlnet_id': Key('str'),
        'gradient_checkpointing': Key('bool'),
        'vae_decoder': Key('str', nullable=True),   # fine-tuned SD-VAE decoder (.pt from tools/finetune_vae_decoder.py); null = stock
    },
    'codebook': {
        'num_embeddings': Key('int', check=_pos),
        'embed_dim': Key('int', check=_pos),
        'sha256_file': Key('str', nullable=True, check=_hex64, hint='64 lowercase hex characters'),
        'sha256_embedding': Key('str', nullable=True, check=_hex64, hint='64 lowercase hex characters'),
    },
    'adapter': {
        'out_channels': Key('int', check=_pos),
        'norm_groups': Key('int', check=lambda v: v > 0 and 16 % v == 0, hint='must divide 16'),
        'l2_normalise_tokens': Key('bool'),
    },
    'conditioning': {
        'paths': Key('str', ('both', 'edges_only', 'adapter_only')),
        'edge_amplification': Key('float', check=_pos),
        'edge_border_fix': Key('bool'),
        'residual_scale': Key('float', check=_nonneg, hint='0 = ControlNet off (A8 ablation only)'),
        'scale_jitter': Key('list_float', nullable=True, check=_pair_range, hint='[lo, hi] with 0 < lo <= hi'),
        'dropout_all': Key('float', check=_prob),
        'dropout_adapter': Key('float', check=_prob),
        'dropout_edges': Key('float', check=_prob),
    },
    'data': {
        'geometry': Key('str', ('squash',)),
        'photo_size': Key('int', check=lambda v: v == 512, hint='512 (latent 64² and adapter 32²→512² are fixed, NEW-sdunet-7)'),
        'sketch_size': Key('int', check=lambda v: v == 256, hint='256 (CVQ-VAE input)'),
        'flip': Key('float', check=_prob),
        'num_workers': Key('int', check=_nonneg),
        'pin_memory': Key('bool'),
    },
    'prompt': {
        'text': Key('str'),
        'dropout_per_sample': Key('float', check=_prob),
        'negative': Key('str'),
        'negative_style': Key('str'),
        'negative_style_above_cfg': Key('float', check=_nonneg),
        # Attribute-conditioned prompts (user 2026-09-22, D6 amended). The attribute files are built from the GT
        # photos, so enabling this for val/test is an oracle ("witness description") run and is reported as the
        # separate "Ours + attributes" row, never as the like-for-like comparison (F10).
        'attributes': {
            'enabled': Key('bool'),
            'dir': Key('str'),
            'on_missing': Key('str', ('fixed', 'raise')),
        },
    },
    'timesteps': {
        'sampler': Key('str', ('uniform', 'low', 'mixed')),
        'shuffle': Key('bool'),
        'mse_weighting': Key('str', ('min_snr', 'legacy', 'uniform')),
        'gamma': Key('float', check=_pos),
    },
    'aux': {
        'max_t': Key('int', check=_nonneg),
        'snr_weight': Key('bool'),
        'max_decode': Key('int', check=_nonneg),
        'loss_size': Key('int', check=lambda v: v == 256, hint='256 (the scoring scale, P6)'),
        'lpips_vgg': Key('float', check=_nonneg),
        'scharr': Key('float', check=_nonneg),
        'gmsd': Key('float', check=_nonneg),
        'latent_l1_scharr': Key('float', check=_nonneg),
        'id': {
            'net': Key('str', ('ir_se50',)),
            'align': Key('str', ('gt_landmarks_mtcnn',)),
            'max_t': Key('int', check=_nonneg),
            'weight': Key('float', check=_nonneg),
            'weights_sha256': Key('str', nullable=True, check=_hex64, hint='64 lowercase hex characters'),
        },
        'fsim_surrogate': _flag_block(),
        'ms_ssim': _flag_block(),
        # Texture-statistics term (2026-09-24, F56/F57): L1 between the log radially-averaged power spectra of prediction
        # and photo above a cutoff frequency — matches how much fine texture there is, not where (no per-pixel gradients, F46).
        'texture': _flag_block(),
        # v1's two-phase schedule (F51): from `start_step` on, the LPIPS and identity terms are multiplied.
        # start_step 0 = off. Scharr and GMSD are not boostable — they stay at their (zero) weights (F46).
        'boost': {
            'start_step': Key('int', check=_nonneg, hint='0 = off'),
            'lpips': Key('float', check=_nonneg),
            'id': Key('float', check=_nonneg),
        },
    },
    'optim': {
        # Learning-rate restart for a refinement fork (--extend --refine): from `step` a new linear warm-up over `warmup`
        # steps to `peak` x base LR, then cosine to min_lr_ratio at total_steps. step null = off.
        'restart': {
            'step': Key('int', nullable=True, check=_nonneg),
            'warmup': Key('int', check=_nonneg),
            'peak': Key('float', check=_pos),
        },
        'lr_controlnet': Key('float', check=_pos),
        'lr_adapter_and_new_columns': Key('float', check=_pos),
        'weight_decay': Key('float', check=_nonneg),
        'betas': Key('list_float', check=_betas),
        'eps': Key('float', check=_pos),
        'warmup_steps': Key('int', check=_nonneg),
        'schedule': Key('str', ('cosine',)),
        'min_lr_ratio': Key('float', check=_prob),
        'total_steps': Key('int', check=_pos),
        # Length of the warm-up + cosine; null = total_steps. Set below total_steps, training continues past the
        # cosine at the floor LR (min_lr_ratio) — a low-LR fine-tuning tail (limit study, user 2026-09-24).
        'schedule_steps': Key('int', nullable=True, check=_pos),
        'batch': Key('int', check=_pos),
        'grad_clip': Key('float', check=_pos),
    },
    'ema': {
        'decay': Key('float', check=lambda v: 0.0 < v < 1.0),
        'include_buffers': Key('bool'),
        'eval_with_ema': Key('bool'),
    },
    'checkpoint': {
        'every_steps': Key('int', check=_pos),
        'keep_last': Key('int', check=_pos),
        'keep_every': Key('int', nullable=True, check=_pos),   # milestone steps kept forever (outside keep_last); null = none
    },
    'val': {
        'every_steps': Key('int', check=_pos),
        'split': Key('str', ('val',)),
        'seeds': Key('list_int', check=lambda v: len(v) > 0 and all(x >= 0 for x in v)),
        'write_png': Key('bool'),
        'loss_t': Key('list_int', check=_t_list),
        'loss_train_stems': Key('int', check=_nonneg),
        'max_images': Key('int', nullable=True, check=_pos),
    },
    'log': {
        'every_steps': Key('int', check=_pos),
        'tensorboard': Key('bool'),
    },
    'inference': {
        'sampler': Key('str', ('dpmpp_2m_karras', 'ddim', 'unipc', 'euler_a')),
        'steps': Key('int', check=_pos),
        'timestep_spacing': Key('str', ('trailing', 'leading', 'linspace')),
        'eta': Key('float', check=_nonneg),
        'guidance_scale': Key('float', check=lambda v: v >= 1.0, hint='>= 1.0'),
        'control_guidance_end': Key('float', check=lambda v: 0.0 < v <= 1.0),
        'guess_mode': Key('bool'),
        'start': Key('str', ('noise', 'xhat', 'sketch')),   # 'sketch': SDEdit start from the sketch image itself (ablation only)
        'strength': Key('float', check=lambda v: 0.0 < v <= 1.0),
        'precision': Key('str', ('fp16', 'bf16', 'fp32')),
        'batch_size': Key('int', check=_pos),
        'output_size': Key('list_int', check=_pair_pos, hint='[W, H]'),
        'use_ema': Key('bool'),
        # Low-frequency guidance (ILVR, Choi et al. ICCV 2021, applied to latents; 2026-09-25): after each denoising step
        # of the first `until` fraction of the schedule, replace the coarse part (down by `factor`, up again) of the latent
        # with the coarse part of the x̂ anchor latent noised to the same level — keeps x̂'s face shape (identity) while SD
        # draws the fine detail. weight 1 = full replacement. enabled false = off.
        'lowfreq_guide': {
            'enabled': Key('bool'),
            'factor': Key('int', check=_pos),
            'weight': Key('float', check=_nonneg),
            'until': Key('float', check=lambda v: 0.0 < v <= 1.0),
        },
    },
}


# ----------------------------------------------------------------------------- validation
def _coerce(value, key, path):
    if value is None:
        if key.nullable:
            return None
        raise ConfigError('%s: null is not allowed' % path)
    kind = key.kind
    if kind == 'bool':
        if not isinstance(value, bool):
            raise ConfigError('%s: expected true/false, got %r' % (path, value))
        out = value
    elif kind == 'int':
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError('%s: expected an integer, got %r' % (path, value))
        out = value
    elif kind == 'float':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError('%s: expected a number (write floats with a dot, e.g. 1.0e-5), got %r' % (path, value))
        out = float(value)
    elif kind == 'str':
        if not isinstance(value, str):
            raise ConfigError('%s: expected a string, got %r' % (path, value))
        out = value
    elif kind in ('list_int', 'list_float'):
        if not isinstance(value, (list, tuple)):
            raise ConfigError('%s: expected a list, got %r' % (path, value))
        items = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigError('%s: list items must be numbers, got %r' % (path, item))
            if kind == 'list_int':
                if not isinstance(item, int):
                    raise ConfigError('%s: list items must be integers, got %r' % (path, item))
                items.append(item)
            else:
                items.append(float(item))
        out = items
    else:  # pragma: no cover - schema error
        raise ConfigError('%s: unknown schema kind %s' % (path, kind))
    if key.choices is not None and out not in key.choices:
        raise ConfigError('%s: %r not in %s' % (path, out, list(key.choices)))
    if key.check is not None and not key.check(out):
        raise ConfigError('%s: invalid value %r%s' % (path, out, (' (' + key.hint + ')') if key.hint else ''))
    return out


def _validate(node, schema, path):
    if not isinstance(node, dict):
        raise ConfigError('%s: expected a mapping' % (path or '<root>'))
    unknown = sorted(set(node) - set(schema))
    if unknown:
        raise ConfigError('%s: unknown key(s) %s' % (path or '<root>', unknown))
    missing = sorted(set(schema) - set(node))
    if missing:
        raise ConfigError('%s: missing key(s) %s' % (path or '<root>', missing))
    out = {}
    for name, spec in schema.items():
        sub = (path + '.' + name) if path else name
        if isinstance(spec, dict):
            out[name] = _validate(node[name], spec, sub)
        else:
            out[name] = _coerce(node[name], spec, sub)
    return out


def _cross_checks(cfg):
    c = cfg['conditioning']
    if c['dropout_all'] + c['dropout_adapter'] + c['dropout_edges'] > 1.0:
        raise ConfigError('conditioning: dropout_all + dropout_adapter + dropout_edges must be <= 1')
    a = cfg['aux']
    for name, block_t in (('aux.id.max_t', a['id']['max_t']),
                          ('aux.fsim_surrogate.max_t', a['fsim_surrogate']['max_t']),
                          ('aux.ms_ssim.max_t', a['ms_ssim']['max_t'])):
        if block_t > a['max_t']:
            raise ConfigError('%s (%d) must be <= aux.max_t (%d): those rows are never decoded' % (name, block_t, a['max_t']))
    o = cfg['optim']
    if o['warmup_steps'] >= o['total_steps']:
        raise ConfigError('optim.warmup_steps must be < optim.total_steps')
    if o['schedule_steps'] is not None and not o['warmup_steps'] < o['schedule_steps'] <= o['total_steps']:
        raise ConfigError('optim.schedule_steps (%d) must be > warmup_steps and <= total_steps (%d)'
                          % (o['schedule_steps'], o['total_steps']))
    if cfg['tone']['enabled'] and cfg['inference']['start'] != 'xhat':
        raise ConfigError("tone.enabled needs inference.start: xhat — there is no anchor to normalise otherwise")
    an = cfg['anchor']
    # t_min >= 300 keeps the target factor sqrt(a/(1-a)) below ~1.3 on SD's schedule; at small t it explodes.
    if an['t_min'] < 300 or not an['t_min'] < an['t_max'] <= 1000:
        raise ConfigError('anchor: need 300 <= t_min < t_max <= 1000 (got %d, %d)' % (an['t_min'], an['t_max']))
    # (anchor.enabled needs inference.start: xhat only while *training* — that check lives in train_controlnet.py.
    # A checkpoint trained with the anchor may still be sampled from noise, e.g. for the attributes row.)
    b = a['boost']
    if b['start_step'] > 0 and b['start_step'] >= o['total_steps']:
        raise ConfigError('aux.boost.start_step (%d) must be < optim.total_steps (%d)'
                          % (b['start_step'], o['total_steps']))
    ds = cfg['run']['dataset']
    if tuple(cfg['inference']['output_size']) != OUTPUT_SIZE[ds]:
        raise ConfigError('inference.output_size %s does not match the scoring geometry of %s %s (D15, MC3)'
                          % (cfg['inference']['output_size'], ds, list(OUTPUT_SIZE[ds])))


def validate(raw):
    """Return a validated deep copy of `raw` (without the resolved-path block)."""
    raw = copy.deepcopy(raw)
    raw.pop(CONFIG_SOURCE_KEY, None)
    cfg = _validate(raw, SCHEMA, '')
    _cross_checks(cfg)
    return cfg


# ----------------------------------------------------------------------------- overrides
def parse_overrides(items):
    """['a.b=1', 'c=[0.8, 1.2]'] -> [('a.b', 1), ('c', [0.8, 1.2])] (values parsed as YAML)."""
    out = []
    for item in items or []:
        if '=' not in item:
            raise ConfigError('--set expects key=value, got %r' % item)
        key, value = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ConfigError('--set: empty key in %r' % item)
        out.append((key, yaml.safe_load(value)))
    return out


def apply_overrides(raw, overrides, allowed_prefixes=None):
    raw = copy.deepcopy(raw)
    for key, value in overrides:
        if allowed_prefixes is not None and not any(key == p or key.startswith(p + '.') for p in allowed_prefixes):
            raise ConfigError('override %s is not allowed here (allowed: %s)' % (key, list(allowed_prefixes)))
        node = raw
        parts = key.split('.')
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError('override %s: unknown key' % key)
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError('override %s: unknown key' % key)
        node[parts[-1]] = value
    return raw


# ----------------------------------------------------------------------------- paths
def project_root():
    return os.path.abspath(os.environ.get('SKETCH_PROJECT_ROOT') or DEFAULT_PROJECT_ROOT)


def resolve_paths(cfg):
    root = project_root()
    p = cfg['paths']

    def under(base, rel):
        return os.path.normpath(rel if os.path.isabs(rel) else os.path.join(base, rel))

    data_root = os.path.abspath(os.environ['SKETCH_DATA_ROOT']) if os.environ.get('SKETCH_DATA_ROOT') \
        else under(root, p['data_root'])
    outputs_root = os.path.abspath(os.environ['SKETCH_OUTPUTS_ROOT']) if os.environ.get('SKETCH_OUTPUTS_ROOT') \
        else under(root, p['outputs_root'])
    ds = cfg['run']['dataset']
    codebook_dir = os.path.join(data_root, 'codebooks', ds + '_cvq')
    return {
        'project_root': root,
        'data_root': data_root,
        'outputs_root': outputs_root,
        'dataset_dir': os.path.join(data_root, 'datasets', ds),
        'splits_dir': os.path.join(data_root, 'splits', ds),
        'codebook_dir': codebook_dir,
        'codebook_file': os.path.join(codebook_dir, 'netG_a_model.pth'),
        'hf_home': under(data_root, p['hf_home']),
        'torch_home': under(data_root, p['torch_home']),
        'ir_se50': under(data_root, p['ir_se50']),
        'landmarks': under(outputs_root, p['landmarks']),
        'attributes_dir': os.path.join(under(data_root, cfg['prompt']['attributes']['dir']), ds),
        'tone_stats_dir': os.path.join(under(data_root, cfg['tone']['dir']), ds),
        'dataset_outputs': os.path.join(outputs_root, ds),
    }


def run_dirs(cfg, run_id):
    base = cfg[CONFIG_SOURCE_KEY]['dataset_outputs']
    return {
        'checkpoints': os.path.join(base, 'checkpoints', run_id),
        'logs': os.path.join(base, 'logs', run_id),
        'val': os.path.join(base, 'val', run_id),
    }


def apply_env(cfg):
    """Point HF_HOME / TORCH_HOME at data/pretrained unless the session already set them; record the result."""
    r = cfg[CONFIG_SOURCE_KEY]
    os.environ.setdefault('HF_HOME', r['hf_home'])
    os.environ.setdefault('TORCH_HOME', r['torch_home'])
    r['env'] = {'HF_HOME': os.environ['HF_HOME'], 'TORCH_HOME': os.environ['TORCH_HOME']}
    return r['env']


# Blocks added after the first runs; filled with their defaults so a config saved by an older checkpoint still
# loads (the defaults reproduce that older behaviour exactly).
BACK_COMPAT_DEFAULTS = {
    ('prompt', 'attributes'): {'enabled': False, 'dir': 'attributes', 'on_missing': 'fixed'},
    ('tone',): {'enabled': False, 'dir': 'tone_stats', 'stats_split': 'train_dev', 'method': 'lab_moments',
                'strength': 1.0, 'target': 'photo'},
    ('anchor',): {'enabled': False, 'prob': 0.6, 't_min': 400, 't_max': 850},
    ('aux', 'boost'): {'start_step': 0, 'lpips': 1.0, 'id': 1.0},
    ('optim', 'schedule_steps'): None,
    ('model', 'vae_decoder'): None,
    ('checkpoint', 'keep_every'): None,
    ('aux', 'texture'): {'enabled': False, 'max_t': 300, 'weight': 0.0},
    ('optim', 'restart'): {'step': None, 'warmup': 0, 'peak': 1.0},
    ('inference', 'lowfreq_guide'): {'enabled': False, 'factor': 4, 'weight': 1.0, 'until': 0.6},
}


def fill_back_compat(raw):
    for path, default in BACK_COMPAT_DEFAULTS.items():
        node = raw
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and path[-1] not in node:
            node[path[-1]] = dict(default) if isinstance(default, dict) else default
    return raw


# ----------------------------------------------------------------------------- entry points
def load_config(path, overrides=None, allowed_prefixes=None):
    """Load, override, validate and resolve. `overrides` is the raw list of 'key=value' strings."""
    with open(path, 'r') as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError('%s is empty' % path)
    raw.pop(CONFIG_SOURCE_KEY, None)
    fill_back_compat(raw)
    parsed = parse_overrides(overrides)
    raw = apply_overrides(raw, parsed, allowed_prefixes)
    cfg = validate(raw)
    cfg[CONFIG_SOURCE_KEY] = resolve_paths(cfg)
    cfg[CONFIG_SOURCE_KEY]['config_file'] = os.path.abspath(path)
    cfg[CONFIG_SOURCE_KEY]['overrides'] = ['%s=%s' % (k, json.dumps(v)) for k, v in parsed]
    return cfg


def plain(cfg):
    """The validated config without the resolved block (what a run's behaviour depends on)."""
    out = copy.deepcopy(cfg)
    out.pop(CONFIG_SOURCE_KEY, None)
    return out


def dump_config(cfg, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        yaml.safe_dump(copy.deepcopy(cfg), fh, sort_keys=False, default_flow_style=None, allow_unicode=True)
    os.replace(tmp, path)


def config_diff(a, b, prefix=''):
    """List of 'key: a -> b' strings for two plain configs."""
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        path = (prefix + '.' + k) if prefix else k
        va, vb = a.get(k, '<missing>'), b.get(k, '<missing>')
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(config_diff(va, vb, path))
        elif va != vb:
            out.append('%s: %r -> %r' % (path, va, vb))
    return out


def default_run_id(cfg):
    return '%s_v2_%s' % (cfg['run']['dataset'], time.strftime('%Y%m%d-%H%M%S'))


# ----------------------------------------------------------------------------- hashing
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def code_sha256(root=CODE_ROOT):
    """sha256 of every .py / .yaml file under 2_controlnet (no git: this is the code record, rule 7)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__' and not d.startswith('.'))
        for name in sorted(filenames):
            if name.endswith(('.py', '.yaml', '.yml', '.txt')):
                full = os.path.join(dirpath, name)
                out[os.path.relpath(full, root)] = sha256_file(full)
    return out


def natural_key(stem):
    return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)
