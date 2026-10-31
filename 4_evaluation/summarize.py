"""Mean over seeds of the scored runs — the numbers of the paper's main table.

    python 4_evaluation/summarize.py --dataset fs2k --runs fs2k_final_seed0 fs2k_final_seed1 fs2k_final_seed2

Reads results/<dataset>/test/<method>/<run_id>/{summary.json, realism.json} written by evaluate.py and realism.py.
"""
import argparse
import json
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS = ('fid', 'kid', 'lpips', 'fsim', 'id', 'brisque', 'clipiqa')
FMT = {'fid': '%.2f', 'kid': '%.4f', 'lpips': '%.4f', 'fsim': '%.4f', 'id': '%.4f', 'brisque': '%.2f', 'clipiqa': '%.3f'}


def metrics(d):
    s = json.load(open(os.path.join(d, 'summary.json')))
    r = json.load(open(os.path.join(d, 'realism.json')))['means']
    return {'fid': s['fid'], 'kid': s['kid']['mean'], 'lpips': s['lpips']['vgg'], 'fsim': s['fsim'], 'id': s['id']['mean'],
            'brisque': r['brisque'], 'clipiqa': r['clipiqa']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--method', default='ours')
    ap.add_argument('--split', default='test')
    ap.add_argument('--results_root', default=os.path.join(EVAL_DIR, 'results'))
    args = ap.parse_args()
    rows = [metrics(os.path.join(args.results_root, args.dataset, args.split, args.method, r)) for r in args.runs]
    print('| run | ' + ' | '.join(k.upper() for k in KEYS) + ' |')
    for name, m in list(zip(args.runs, rows)) + [('mean', {k: sum(m[k] for m in rows) / len(rows) for k in KEYS})]:
        print('| %s | ' % name + ' | '.join(FMT[k] % m[k] for k in KEYS) + ' |')


if __name__ == '__main__':
    main()
