"""No-reference realism scores — SUPPLEMENTARY, not part of eval-v1 (user 2026-09-24).

BRISQUE (lower = more natural; Mittal et al. 2012; natural-scene statistics) and CLIP-IQA (higher = better; Wang et
al. AAAI 2023; CLIP RN50 "Good photo." vs "Bad photo."), both from piq 0.8.0. Neither metric looks at the GT photo, so
they measure how photo-like an output is, not whether it is the right person. The eval-v1 files are not touched;
rows go to `RUNS_realism.md`, not `RUNS.md`.

Geometry = eval-v1: RGB, PIL-bicubic squash to the scoring size, 8-bit PNG, before scoring.
Weights (downloaded 2026-09-24 with the user's approval): `data/pretrained/clip/{RN50.pt,clipiqa_tokens.pt}` (piq reads
`~/.cache/clip`, a symlink to it) and `data/pretrained/torch_cache/hub/checkpoints/brisque_svm_weights.pt`.

    python 4_evaluation/realism.py --dataset wildsketch --split test --method ours --run_id <id> --pred_dir <folder>
    python 4_evaluation/realism.py --dataset wildsketch --split test --method gt --run_id gt-photos --gt   # reference row
    python 4_evaluation/realism.py --dataset wildsketch --split test --compare decp/<a>[,<a2>] ours/<b>[,<b2>] [--out f.json]
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
import protocol as P  # noqa: E402
import geometry as G  # noqa: E402

os.environ.setdefault('TORCH_HOME', P.TORCH_CACHE)
METRICS = ('brisque', 'clipiqa')
LOWER_IS_BETTER = ('brisque',)


def run_dir(dataset, split, method, run_id):
    return os.path.join(EVAL_DIR, 'results', dataset, split, method, run_id)


def score(files, device, batch=32):
    import torch
    import piq
    from PIL import Image
    clipiqa = piq.CLIPIQA(data_range=1.).to(device)
    out = {'brisque': [], 'clipiqa': []}
    for i in range(0, len(files), batch):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) / 255.)
                         .permute(2, 0, 1) for f in files[i:i + batch]]).to(device)
        with torch.no_grad():
            out['brisque'] += piq.brisque(x, data_range=1., reduction='none').flatten().tolist()
            out['clipiqa'] += clipiqa(x).flatten().tolist()
    return out


def cmd_score(args):
    gt = G.gt_index(args.dataset, args.split)
    stems = sorted(gt, key=lambda s: int(s) if s.isdigit() else s)
    if args.gt:
        src = gt
    else:
        pred = G.index_by_stem(args.pred_dir)
        G.pair(gt, pred)
        src = pred
    size = P.SCORING_SIZE[args.dataset]
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for s in stems:
            dst = os.path.join(tmp, s + '.png')
            G.prepare(src[s], dst, size)
            files.append(dst)
        t0 = time.time()
        vals = score(files, args.device)
    out_dir = run_dir(args.dataset, args.split, args.method, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'realism.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stem'] + list(METRICS))
        for i, s in enumerate(stems):
            w.writerow([s] + [vals[m][i] for m in METRICS])
    summary = {'dataset': args.dataset, 'split': args.split, 'method': args.method, 'run_id': args.run_id,
               'n_images': len(stems), 'date': time.strftime('%Y-%m-%d %H:%M:%S'),
               'source': 'GT photos' if args.gt else os.path.abspath(args.pred_dir),
               'means': {m: float(np.mean(vals[m])) for m in METRICS},
               'std_over_images': {m: float(np.std(vals[m])) for m in METRICS},
               'seconds': time.time() - t0, 'piq': __import__('piq').__version__}
    json.dump(summary, open(os.path.join(out_dir, 'realism.json'), 'w'), indent=2)
    log = os.path.join(EVAL_DIR, 'RUNS_realism.md')
    if not os.path.isfile(log):
        with open(log, 'w') as fh:
            fh.write('# Supplementary realism runs — append-only (BRISQUE ↓, CLIP-IQA ↑; piq 0.8.0; eval-v1 geometry)\n\n'
                     'No-reference metrics: they do not compare with the GT photo. Never a headline number.\n\n'
                     '| run_id | date | dataset | split | method | N | BRISQUE ↓ | CLIP-IQA ↑ | notes |\n'
                     '|---|---|---|---|---|---|---|---|---|\n')
    with open(log, 'a') as fh:
        fh.write('| %s | %s | %s | %s | %s | %d | %.3f | %.4f | %s |\n' % (
            args.run_id, summary['date'][:10], args.dataset, args.split, args.method, len(stems),
            summary['means']['brisque'], summary['means']['clipiqa'], args.notes or ''))
    print('%s  BRISQUE %.3f  CLIP-IQA %.4f  (N=%d)' % (args.run_id, summary['means']['brisque'],
                                                       summary['means']['clipiqa'], len(stems)))


def load_per_image(dataset, split, spec):
    method, run_ids = spec.split('/', 1)
    acc = {}
    ids = run_ids.split(',')
    for rid in ids:
        with open(os.path.join(run_dir(dataset, split, method, rid), 'realism.csv')) as fh:
            for row in csv.DictReader(fh):
                d = acc.setdefault(row['stem'], {m: 0.0 for m in METRICS})
                for m in METRICS:
                    d[m] += float(row[m]) / len(ids)
    return acc


def cmd_compare(args):
    import compare as C
    a = load_per_image(args.dataset, args.split, args.compare[0])
    b = load_per_image(args.dataset, args.split, args.compare[1])
    saved = C.LOWER_IS_BETTER
    C.LOWER_IS_BETTER = LOWER_IS_BETTER
    try:
        rows = [C.compare(a, b, m) for m in METRICS]
    finally:
        C.LOWER_IS_BETTER = saved
    adj = C.holm([r['p'] if r['p'] is not None else 1.0 for r in rows])
    lines = ['%s realism — A = %s   B = %s' % (args.split, args.compare[0], args.compare[1]),
             'metric      n    mean A    mean B   median Δ     rbc          p   p (Holm)  verdict']
    for r, pa in zip(rows, adj):
        r['p_holm'] = pa
        verdict = ('%s, significant' % r['direction']) if pa < 0.05 else 'no significant difference'
        lines.append('%-9s %4d  %8.4f  %8.4f  %9.4f  %6.3f  %9.2e  %9.2e  %s' % (
            r['metric'], r['n'], r['mean_a'], r['mean_b'], r['median_delta'], r['rbc'], r['p'], pa, verdict))
    print('\n'.join(lines))
    if args.out:
        json.dump({'a': args.compare[0], 'b': args.compare[1], 'rows': rows}, open(args.out, 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--method')
    ap.add_argument('--run_id')
    ap.add_argument('--pred_dir')
    ap.add_argument('--gt', action='store_true', help='score the GT photos themselves (reference row)')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    ap.add_argument('--out')
    ap.add_argument('--notes', default='')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args)
    else:
        if not (args.method and args.run_id and (args.pred_dir or args.gt)):
            ap.error('--method, --run_id and --pred_dir (or --gt) are required to score')
        cmd_score(args)


if __name__ == '__main__':
    main()
