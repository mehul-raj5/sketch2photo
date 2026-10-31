"""Score one folder of generated images against a dataset split — the only source of reported numbers (rule 7).

    python 4_evaluation/evaluate.py --dataset wildsketch --split test --method decp \
        --run_id decp-wildsketch-official-s0 \
        --pred_dir DECP/outputs/wildsketch/test/decp-wildsketch-official-s0 --device cuda:0

Writes results/<dataset>/<split>[-<variant>]/<method>/<run_id>/{per_image.csv, summary.json} and appends one row to
RUNS.md (unless --no_runs_md, for self-tests and diagnostics). Metrics: README §8 / CLAUDE.md protocol.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
import protocol as P  # noqa: E402

os.environ.setdefault('TORCH_HOME', P.TORCH_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import geometry  # noqa: E402

SCORING_FILES = ('protocol.py', 'geometry.py', 'fsim.py', 'lpips_score.py', 'fid.py', 'id_score.py',
                 'supplementary.py', 'evaluate.py')
PER_IMAGE_COLUMNS = ('stem', 'lpips_alex', 'lpips_squeeze', 'lpips_vgg', 'fsim', 'id', 'id_skipped', 'ssim', 'psnr')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Score generated images (4_evaluation protocol)')
    p.add_argument('--dataset', required=True, choices=sorted(P.SCORING_SIZE))
    p.add_argument('--split', required=True, choices=['test', 'train', 'val', 'train_dev'])
    p.add_argument('--method', required=True)
    p.add_argument('--run_id', required=True)
    p.add_argument('--pred_dir', required=True)
    p.add_argument('--device', required=True, help='e.g. cuda:0 (never defaulted)')
    p.add_argument('--variant', default=None, help='T6 degradation variant, e.g. blur2')
    p.add_argument('--kid_subset_size', type=int, default=None, help='also KID over paired random subsets of this size')
    p.add_argument('--kid_subset_count', type=int, default=200)
    p.add_argument('--notes', default='')
    p.add_argument('--no_runs_md', action='store_true', help='do not append to RUNS.md (self-tests, diagnostics)')
    p.add_argument('--results_root', default=os.path.join(EVAL_DIR, 'results'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--keep_work', action='store_true', help='keep the prepared PNGs under work/')
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args(argv)


def pred_provenance(pred_dir):
    path = os.path.join(pred_dir, 'run.json')
    if not os.path.isfile(path):
        return {'run_json': None, 'checkpoint_sha256': None, 'seed': None}
    with open(path) as fh:
        run = json.load(fh)
    ckpt = None
    if isinstance(run.get('checkpoint'), dict):
        ckpt = run['checkpoint'].get('sha256')
    extra = run.get('extra') or {}
    ckpt = ckpt or extra.get('ema_sha256') or extra.get('trainable_sha256')
    return {'run_json': path, 'run_json_sha256': sha256_file(path), 'checkpoint_sha256': ckpt, 'seed': run.get('seed')}


def mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def fmt(v, digits):
    return '—' if v is None else ('%.*f' % (digits, v))


def append_runs_md(summary, notes):
    m = summary['means']
    lp = summary['lpips']
    if P.HEADLINE_LPIPS:
        lp_cell = '%.4f (%s)' % (lp[P.HEADLINE_LPIPS], P.HEADLINE_LPIPS)
    else:
        lp_cell = 'a %.4f · s %.4f · v %.4f' % (lp['alex'], lp['squeeze'], lp['vgg'])
    ckpt = summary['provenance'].get('checkpoint_sha256')
    seed = summary['provenance'].get('seed')
    split = summary['split'] + ('-' + summary['variant'] if summary['variant'] else '')
    row = '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%d) | %s |\n' % (
        summary['run_id'], summary['date'][:10], summary['protocol'], summary['dataset'], split, summary['method'],
        ckpt[:12] if ckpt else '—', '—' if seed is None else seed, fmt(summary['fid'], 2), lp_cell,
        fmt(m['fsim'], 4), fmt(m['id'], 3), summary['id']['skipped'],
        ('N=%d. ' % summary['n_images']) + (notes or ''))
    with open(os.path.join(EVAL_DIR, 'RUNS.md'), 'a') as fh:
        fh.write(row)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    size = P.SCORING_SIZE[args.dataset]
    split_name = args.split + ('-' + args.variant if args.variant else '')
    out_dir = os.path.join(args.results_root, args.dataset, split_name, args.method, args.run_id)
    if os.path.exists(os.path.join(out_dir, 'summary.json')) and not args.overwrite:
        raise SystemExit('%s already scored — pass --overwrite' % out_dir)

    gt = geometry.gt_index(args.dataset, args.split)
    pred = geometry.index_by_stem(os.path.abspath(args.pred_dir))
    stems = geometry.pair(gt, pred)

    work = os.path.join(EVAL_DIR, 'work', '%s_%d' % (args.run_id, os.getpid()))
    gt_dir, pred_dir = os.path.join(work, 'gt'), os.path.join(work, 'pred')
    timings = {}
    t = time.time()
    gt_resized = geometry.prepare_folder(gt, stems, gt_dir, size)
    pred_resized = geometry.prepare_folder(pred, stems, pred_dir, size)
    timings['prepare'] = time.time() - t
    if pred_resized:
        print('WARNING: %d predictions were not at the scoring size %s and were squashed (PIL bicubic), e.g. %s'
              % (len(pred_resized), size, pred_resized[:5]))

    import fsim
    from fid import InceptionFeatures, fid_from_features, kid_from_features, kid_paired_subsets
    from id_score import IDScorer
    from lpips_score import LPIPSScorer
    from supplementary import ssim_psnr

    if device.type == 'cuda':
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    rows = {s: {'stem': s} for s in stems}
    pp = {s: os.path.join(pred_dir, s + '.png') for s in stems}
    gp = {s: os.path.join(gt_dir, s + '.png') for s in stems}

    t = time.time()
    lp = LPIPSScorer(device)
    for s in stems:
        for net, v in lp.score(pp[s], gp[s]).items():
            rows[s]['lpips_' + net] = v
    del lp
    timings['lpips'] = time.time() - t

    t = time.time()
    for s in stems:
        rows[s]['fsim'] = fsim.fsim_files(pp[s], gp[s])
    timings['fsim'] = time.time() - t

    t = time.time()
    ids = IDScorer(device)
    id_gt = gt if P.ID_GT_NATIVE else gp
    for s in stems:
        score, why = ids.score(pp[s], id_gt[s])
        rows[s]['id'] = score
        rows[s]['id_skipped'] = why
    id_meta = {'weights_sha256': ids.weights_sha256, 'mtcnn_sha256': ids.mtcnn_sha256}
    del ids
    timings['id'] = time.time() - t

    t = time.time()
    for s in stems:
        rows[s]['ssim'], rows[s]['psnr'] = ssim_psnr(pp[s], gp[s])
    timings['ssim_psnr'] = time.time() - t

    t = time.time()
    feats = InceptionFeatures(device, num_workers=args.num_workers)
    f_pred = feats([pp[s] for s in stems])
    f_gt = feats([gp[s] for s in stems])
    fid_value = fid_from_features(f_pred, f_gt)
    kid = kid_from_features(f_pred, f_gt)
    kid_sub = None
    if args.kid_subset_size:
        kid_sub = kid_paired_subsets(f_pred, f_gt, args.kid_subset_size, args.kid_subset_count)
    timings['fid_kid'] = time.time() - t

    per_image = [rows[s] for s in stems]
    skipped = [r['stem'] for r in per_image if r['id_skipped']]
    means = {c: mean([r.get(c) for r in per_image]) for c in PER_IMAGE_COLUMNS if c not in ('stem', 'id_skipped')}
    summary = {
        'protocol': P.PROTOCOL, 'dataset': args.dataset, 'split': args.split, 'variant': args.variant,
        'method': args.method, 'run_id': args.run_id, 'n_images': len(stems), 'scoring_size': list(size),
        'fid': fid_value, 'kid': kid, 'kid_paired_subsets': kid_sub,
        'lpips': {n: means['lpips_' + n] for n in P.LPIPS_NETS}, 'lpips_headline': P.HEADLINE_LPIPS,
        'fsim': means['fsim'],
        'id': {'mean': means['id'], 'n_scored': len(stems) - len(skipped), 'skipped': len(skipped),
               'skipped_stems': skipped, 'gt_geometry': 'native' if P.ID_GT_NATIVE else 'scoring size', **id_meta},
        'means': means,
        'resized': {'gt': len(gt_resized), 'pred': len(pred_resized), 'pred_stems': pred_resized},
        'pred_dir': os.path.abspath(args.pred_dir), 'gt_split_dir': os.path.dirname(gt[stems[0]]),
        'provenance': pred_provenance(os.path.abspath(args.pred_dir)),
        'timings_seconds': timings,
        'peak_memory_gb': (torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == 'cuda' else None,
        'scoring_files_sha256': {f: sha256_file(os.path.join(EVAL_DIR, f)) for f in SCORING_FILES},
        'environment': environment(),
        'command': ' '.join([sys.executable] + (argv if argv is not None else sys.argv)),
        'device': str(device), 'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'per_image.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PER_IMAGE_COLUMNS)
        w.writeheader()
        for r in per_image:
            w.writerow({c: ('' if r.get(c) is None else r.get(c)) for c in PER_IMAGE_COLUMNS})
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    if not args.no_runs_md:
        append_runs_md(summary, args.notes)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    print('%s  N=%d  FID %.2f  KID %.5f  LPIPS alex %.4f squeeze %.4f vgg %.4f  FSIM %.4f  ID %s (skipped %d)  '
          'SSIM %.4f  PSNR %.2f' % (args.run_id, len(stems), fid_value, kid['mean'], summary['lpips']['alex'],
                                    summary['lpips']['squeeze'], summary['lpips']['vgg'], summary['fsim'],
                                    fmt(means['id'], 4), len(skipped), means['ssim'], means['psnr']))
    if kid_sub:
        print('KID over %d paired %d-image subsets: %.5f ± %.5f' % (kid_sub['subsets'], kid_sub['subset_size'],
                                                                   kid_sub['mean'], kid_sub['std']))
    print('timings (s): %s' % {k: round(v, 1) for k, v in timings.items()})
    print('wrote %s' % out_dir)
    return summary


def environment():
    import importlib
    env = {'python': sys.version.split()[0], 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ('numpy', 'scipy', 'PIL', 'lpips', 'pytorch_fid', 'skimage', 'cv2', 'torchvision'):
        try:
            env[mod] = getattr(importlib.import_module(mod), '__version__', 'unknown')
        except ImportError:
            env[mod] = None
    return env


if __name__ == '__main__':
    main()
