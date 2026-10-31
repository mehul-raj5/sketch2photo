"""Paired significance test between two scored runs (`4_evaluation/README.md` §1, T2).

FID and KID are distribution-level statistics over a whole folder: there is no per-image value, so they
cannot be tested pairwise and are not reported here. LPIPS, FSIM, ID, SSIM and PSNR are per-image, and
because both runs are scored on the *same* stems the comparison is paired — which is far more sensitive
than comparing two means with independent-sample tests.

Test: two-sided Wilcoxon signed-rank on the per-image differences (`pred_b - pred_a`), with `zero_method
= 'wilcox'` (exact ties dropped, the convention `scipy` documents for this test). Reported alongside it:

  * `n`            — pairs actually used, after dropping ties and rows either run could not score;
  * `median_delta` — median of the paired differences, in the metric's own units;
  * `rbc`          — matched-pairs rank-biserial correlation, the effect size for this test
                     (`(W+ - W-) / (W+ + W-)`, in [-1, 1]); a p-value alone says nothing about size.
                     Its sign follows the direction of `b - a`, not of "better": for a
                     lower-is-better metric a strong improvement gives rbc near -1. Read the
                     `direction` field for quality;
  * `p_holm`       — Holm-Bonferroni adjusted p across the metrics tested in one call, because testing
                     four metrics and quoting the smallest raw p is how false positives get into papers.

Multiple seeds (D-randomness: our method is reported over 3 seeds) are handled by passing a run id per
seed; the per-image values are averaged over seeds first, so each stem still contributes exactly one
pair and the test stays valid.

    python 4_evaluation/compare.py --dataset wildsketch --split test \
        --a decp/decp-wildsketch-official-s0 --b ours/ws_v2_final_seed0,ours/ws_v2_final_seed1 \
        [--metrics lpips_vgg,fsim,id] [--out results/comparisons/<name>.json]

`--a` and `--b` are `<method>/<run_id>` as they appear under `results/<dataset>/<split>/`; several
comma-separated run ids mean seeds of the same run. Read-only: this script never rescores anything and
never writes into a run's folder, so `eval-v1` is untouched.
"""
import argparse
import csv
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
import protocol as P                                                  # noqa: E402

DEFAULT_METRICS = ('lpips_' + P.HEADLINE_LPIPS, 'fsim', 'id')
LOWER_IS_BETTER = ('lpips_alex', 'lpips_squeeze', 'lpips_vgg')


def load_per_image(results_root, dataset, split, spec):
    """'<method>/<run_id>[,<run_id>...]' → {stem: {metric: value}}, averaged over the given seeds."""
    method, _, run_ids = spec.partition('/')
    if not run_ids:
        raise SystemExit('--a/--b must be <method>/<run_id>, got %r' % spec)
    per_seed = []
    for run_id in run_ids.split(','):
        path = os.path.join(results_root, dataset, split, method, run_id, 'per_image.csv')
        if not os.path.isfile(path):
            raise SystemExit('no scored run at %s — score it with evaluate.py first' % path)
        table = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                values = {}
                for key, raw in row.items():
                    if key == 'stem' or raw is None or raw == '':
                        continue
                    try:
                        values[key] = float(raw)
                    except ValueError:
                        continue                                      # id_skipped and other non-numeric flags
                table[row['stem']] = values
        per_seed.append(table)

    stems = set(per_seed[0])
    for table in per_seed[1:]:
        stems &= set(table)
    merged = {}
    for stem in stems:
        values = {}
        for metric in per_seed[0][stem]:
            present = [t[stem][metric] for t in per_seed if metric in t[stem]]
            if len(present) == len(per_seed):                          # only average a metric every seed has
                values[metric] = sum(present) / len(present)
        merged[stem] = values
    return method, run_ids.split(','), merged


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment, order preserved, monotonicity enforced."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        value = (len(pvalues) - rank) * pvalues[i]
        running = max(running, min(1.0, value))
        adjusted[i] = running
    return adjusted


def compare(a_values, b_values, metric):
    """Paired Wilcoxon signed-rank on the stems both runs scored for this metric."""
    from scipy.stats import wilcoxon

    stems = sorted(s for s in a_values if s in b_values
                   and metric in a_values[s] and metric in b_values[s])
    deltas = [b_values[s][metric] - a_values[s][metric] for s in stems]
    non_zero = [d for d in deltas if d != 0.0]
    row = {'metric': metric, 'n_pairs_available': len(stems), 'n': len(non_zero),
           'mean_a': (sum(a_values[s][metric] for s in stems) / len(stems)) if stems else None,
           'mean_b': (sum(b_values[s][metric] for s in stems) / len(stems)) if stems else None,
           'lower_is_better': metric in LOWER_IS_BETTER}
    if len(non_zero) < 6:                                             # below this the exact test is meaningless
        row.update({'median_delta': None, 'statistic': None, 'p': None, 'rbc': None,
                    'note': 'too few non-tied pairs (%d) for a Wilcoxon test' % len(non_zero)})
        return row

    ordered = sorted(non_zero)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])

    result = wilcoxon(non_zero, zero_method='wilcox', alternative='two-sided')
    # Rank-biserial correlation from the two signed-rank sums; scipy returns min(W+, W-).
    w_plus = float(wilcoxon(non_zero, zero_method='wilcox', alternative='greater').statistic)
    total = len(non_zero) * (len(non_zero) + 1) / 2.0
    row.update({'median_delta': median, 'statistic': float(result.statistic), 'p': float(result.pvalue),
                'rbc': (2.0 * w_plus - total) / total})
    better = (median < 0) if row['lower_is_better'] else (median > 0)
    row['direction'] = 'b better' if better else ('tied' if median == 0 else 'a better')
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=sorted(P.SCORING_SIZE))
    ap.add_argument('--split', required=True, choices=['test', 'train', 'val', 'train_dev'])
    ap.add_argument('--a', required=True, help='baseline run, <method>/<run_id>[,<run_id>...]')
    ap.add_argument('--b', required=True, help='run under test, <method>/<run_id>[,<run_id>...]')
    ap.add_argument('--metrics', default=','.join(DEFAULT_METRICS))
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--results_root', default=os.path.join(EVAL_DIR, 'results'))
    ap.add_argument('--out', default=None, help='write the full result as JSON here')
    args = ap.parse_args()

    a_method, a_runs, a_values = load_per_image(args.results_root, args.dataset, args.split, args.a)
    b_method, b_runs, b_values = load_per_image(args.results_root, args.dataset, args.split, args.b)
    shared = sorted(set(a_values) & set(b_values))
    if not shared:
        raise SystemExit('the two runs share no stems — are they the same dataset and split?')
    if set(a_values) != set(b_values):
        print('warning: %d stems in A and %d in B; testing the %d they share'
              % (len(a_values), len(b_values), len(shared)), file=sys.stderr)

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    rows = [compare(a_values, b_values, m) for m in metrics]
    testable = [r for r in rows if r['p'] is not None]
    for row, p_adj in zip(testable, holm([r['p'] for r in testable])):
        row['p_holm'] = p_adj
        row['significant'] = p_adj < args.alpha

    print('%s %s — A = %s/%s   B = %s/%s   (%d shared stems)'
          % (args.dataset, args.split, a_method, ','.join(a_runs), b_method, ','.join(b_runs), len(shared)))
    print('%-12s %8s %9s %9s %11s %8s %10s %10s  %s'
          % ('metric', 'n', 'mean A', 'mean B', 'median Δ', 'rbc', 'p', 'p (Holm)', 'verdict'))
    for r in rows:
        if r['p'] is None:
            print('%-12s %8d %9s %9s   %s' % (r['metric'], r['n'], '-', '-', r['note']))
            continue
        verdict = ('%s, significant' % r['direction']) if r['significant'] else 'no significant difference'
        print('%-12s %8d %9.4f %9.4f %11.4f %8.3f %10.2e %10.2e  %s'
              % (r['metric'], r['n'], r['mean_a'], r['mean_b'], r['median_delta'], r['rbc'],
                 r['p'], r['p_holm'], verdict))
    print('\nTwo-sided Wilcoxon signed-rank on paired per-image values; Holm-adjusted over %d metrics, '
          'alpha = %g.\nFID and KID are distribution statistics and have no per-image value, so they are '
          'not testable here.' % (len(testable), args.alpha))

    payload = {'protocol': P.PROTOCOL, 'dataset': args.dataset, 'split': args.split, 'alpha': args.alpha,
               'a': {'method': a_method, 'run_ids': a_runs}, 'b': {'method': b_method, 'run_ids': b_runs},
               'n_shared_stems': len(shared), 'test': 'wilcoxon signed-rank, two-sided, zero_method=wilcox',
               'multiplicity': 'holm-bonferroni', 'results': rows}
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(EVAL_DIR, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print('wrote %s' % out)


if __name__ == '__main__':
    main()
