"""Post-hoc stage of the reported results: codebook-detail fusion and, for gray datasets, conversion to luminance.

    Y_out = clip(Y_ours + a * (Y_xhat - G_sigma(Y_xhat)), 0, 255)

computed on the Y channel of YCbCr at the scoring size (PIL bicubic resize), chroma kept from our output; with --gray the
output is written as gray RGB (Y only). `xhat` is the frozen-codebook estimate written by
`generate.py --xhat_only --set tone.enabled=false`. Both inputs come from the sketch alone; no ground truth is read.
The weight `a` per dataset was chosen on the validation split (mean rank over KID, LPIPS, FSIM, ID; ties -> ID):
wildsketch 0.5, fs2k 0.25, cuhk 0.0, cufsf 0.0 (gray). With a = 0 the x̂ folder is not needed.

    python 3_inference/postprocess.py --src outputs/fs2k/test/fs2k_main_seed0 --xhat outputs/fs2k/test/xhat_raw \
        --out outputs/fs2k/test/fs2k_final_seed0 --a 0.25 --size 256x256
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SIGMA = 1.5


def ycc(path, size):
    return np.asarray(Image.open(path).convert('RGB').resize(size, Image.BICUBIC).convert('YCbCr'), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='folder of generated PNGs (one seed)')
    ap.add_argument('--xhat', default=None, help='folder of raw codebook estimates x̂ (required when a > 0)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--a', type=float, required=True, help='fusion weight of the x̂ detail layer')
    ap.add_argument('--size', required=True, help='scoring size WxH, e.g. 256x256 or 200x250')
    ap.add_argument('--gray', action='store_true', help='write luminance only (gray datasets: cufsf)')
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split('x'))
    if args.a and not args.xhat:
        raise SystemExit('--xhat is required when --a > 0')
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.endswith('.png'))
    for f in files:
        o = ycc(os.path.join(args.src, f), size)
        y = o[..., 0]
        if args.a:
            yx = ycc(os.path.join(args.xhat, f), size)[..., 0]
            y = y + args.a * (yx - gaussian_filter(yx, SIGMA, mode='reflect'))
        y = np.clip(y, 0, 255)
        if args.gray:
            Image.fromarray(np.round(y).astype(np.uint8), mode='L').convert('RGB').save(os.path.join(args.out, f))
        else:
            o[..., 0] = y
            Image.fromarray(np.round(o).astype(np.uint8), mode='YCbCr').convert('RGB').save(os.path.join(args.out, f))
    run_json = os.path.join(args.src, 'run.json')
    r = json.load(open(run_json)) if os.path.isfile(run_json) else {}
    r['postprocess'] = {'fusion_a': args.a, 'sigma': SIGMA, 'gray': args.gray, 'source': os.path.basename(os.path.normpath(args.src)),
                        'xhat_raw': args.xhat and os.path.basename(os.path.normpath(args.xhat)), 'size': list(size),
                        'script': '3_inference/postprocess.py'}
    json.dump(r, open(os.path.join(args.out, 'run.json'), 'w'), indent=1)
    print('wrote %d images to %s' % (len(files), args.out))


if __name__ == '__main__':
    main()
