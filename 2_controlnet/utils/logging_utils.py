"""Run logging for v2 (NEW-pipeline-6, M4, M12).

- The step is the optimiser step everywhere (console, JSONL, TensorBoard); TensorBoard purges events after the
  resume step, so a resumed curve has no duplicates.
- metrics.jsonl: one JSON object per log event ({step, kind, time, ...}); kinds train / val / checkpoint.
- TimestepBins: per-100-t-bin means of per-sample losses over a logging interval, plus the t histogram (M4).
- run_record.jsonl: one record per start or resume (argv, environment, code and input sha256, pipeline info).
"""
import json
import math
import os
import platform
import sys
import time

N_BINS = 10
BIN_WIDTH = 100


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _flatten(values, prefix=''):
    out = {}
    for key, value in values.items():
        name = (prefix + '/' + str(key)) if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    out['%s/%d' % (name, i)] = item
        else:
            out[name] = value
    return out


class MeanMeter(object):
    """Means of scalar values over a logging interval; non-finite values are counted, not averaged."""

    def __init__(self):
        self.sums, self.counts, self.nonfinite = {}, {}, {}

    def add(self, scalars):
        for key, value in scalars.items():
            if value is None or isinstance(value, bool):
                continue
            value = float(value)
            if not math.isfinite(value):
                self.nonfinite[key] = self.nonfinite.get(key, 0) + 1
                continue
            self.sums[key] = self.sums.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self):
        out = dict((k, self.sums[k] / self.counts[k]) for k in self.sums)
        for key, count in self.nonfinite.items():
            out['nonfinite_' + key] = count
        return out

    def reset(self):
        self.__init__()


class TimestepBins(object):
    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, timesteps, values):
        ts = timesteps.detach().cpu().long().tolist()
        vs = values.detach().float().cpu().tolist()
        sums = self.sums.setdefault(name, [0.0] * N_BINS)
        counts = self.counts.setdefault(name, [0] * N_BINS)
        for t, v in zip(ts, vs):
            if not math.isfinite(v):
                continue
            b = min(N_BINS - 1, max(0, int(t) // BIN_WIDTH))
            sums[b] += v
            counts[b] += 1

    def summary(self):
        out = {}
        for name in self.sums:
            out[name] = [(s / c) if c else None for s, c in zip(self.sums[name], self.counts[name])]
        if 'mse' in self.counts:
            out['t_hist'] = list(self.counts['mse'])
        return out

    def reset(self):
        self.__init__()


class RunLogger(object):
    def __init__(self, log_dir, tensorboard, start_step=0):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._metrics = open(os.path.join(log_dir, 'metrics.jsonl'), 'a')
        self._text = open(os.path.join(log_dir, 'train.log'), 'a')
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError('log.tensorboard: true needs the `tensorboard` package') from exc
            self.tb = SummaryWriter(os.path.join(log_dir, 'tb'), purge_step=(int(start_step) + 1) if start_step else None)

    def info(self, message):
        line = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), message)
        print(line, flush=True)
        self._text.write(line + '\n')
        self._text.flush()

    def metrics(self, step, kind, values):
        record = {'step': int(step), 'kind': kind, 'time': time.time()}
        record.update(values)
        self._metrics.write(json.dumps(record, sort_keys=True, default=_json_default) + '\n')
        self._metrics.flush()
        if self.tb is not None:
            for key, value in _flatten(values).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    self.tb.add_scalar('%s/%s' % (kind, key), float(value), int(step))
            self.tb.flush()

    def close(self):
        if self.tb is not None:
            self.tb.close()
        self._metrics.close()
        self._text.close()


def append_run_record(log_dir, record):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'run_record.jsonl'), 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True, default=_json_default) + '\n')


def environment_record(device):
    import torch
    rec = {'python': sys.version.split()[0], 'platform': platform.platform(), 'host': platform.node(),
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device),
           'env': dict((k, os.environ.get(k)) for k in ('HF_HOME', 'TORCH_HOME', 'CUDA_VISIBLE_DEVICES',
                                                        'CUBLAS_WORKSPACE_CONFIG'))}
    for name in ('diffusers', 'transformers', 'accelerate', 'lpips', 'piq', 'numpy', 'PIL', 'yaml'):
        try:
            rec[name] = getattr(__import__(name), '__version__', None)
        except Exception:  # noqa: BLE001 - optional package
            rec[name] = None
    if device.type == 'cuda':
        rec['gpu'] = torch.cuda.get_device_name(device)
        rec['gpu_total_gb'] = round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 2)
        rec['cudnn'] = torch.backends.cudnn.version()
    return rec
