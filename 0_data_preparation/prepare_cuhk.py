#!/usr/bin/env python3
"""Build data/datasets/cuhk (CUFS, CUHK student subset: 88 train / 100 test pairs) from the downloaded photos and sketches.

    python 0_data_preparation/prepare_cuhk.py --photos <dir with f-005-01.jpg ...> --sketches <dir with F2-005-01-sz1.jpg ...>

Rules (identical to the dataset used in the paper):
- the pair stem is the photo name (f-005-01, m1-001-01, ...); the sketch name maps to it by lower-casing, dropping "-sz1"
  and turning the "f2-" / "m2-" prefixes into "f-" / "m-";
- split: sketches whose names start with upper-case "F2-" / "M2-" are the training set (88), the others the test set (100);
- files are copied byte for byte (renamed, never re-encoded) to cuhk/{train,test}/{a,b}/<stem>.<ext>
  (a = photo, b = sketch), with ORIGINAL_NAMES.csv mapping every stem back to the downloaded files.
"""
import argparse
import csv
import os
import shutil
import sys

EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def sketch_to_stem(name):
    """Original sketch file name -> (pair stem, split)."""
    base = os.path.splitext(os.path.basename(name))[0]
    split = 'train' if base[:3] in ('F2-', 'M2-') else 'test'
    s = base.lower()
    if s.endswith('-sz1'):
        s = s[:-4]
    for a, b in (('f2-', 'f-'), ('m2-', 'm-')):
        if s.startswith(a):
            s = b + s[len(a):]
    return s, split


def index(folder):
    out = {}
    for dp, _, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(EXT):
                out[os.path.splitext(f)[0]] = os.path.join(dp, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--photos', required=True)
    ap.add_argument('--sketches', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datasets', 'cuhk'))
    args = ap.parse_args()
    photos, sketches = index(args.photos), index(args.sketches)
    rows = []
    for sk_base, sk_path in sorted(sketches.items()):
        stem, split = sketch_to_stem(sk_path)
        if stem not in photos:
            sys.exit('no photo for sketch %s (expected %s.*)' % (sk_path, stem))
        rows.append((stem, split, photos[stem], sk_path))
    n = {s: sum(r[1] == s for r in rows) for s in ('train', 'test')}
    if n != {'train': 88, 'test': 100}:
        sys.exit('expected 88 train / 100 test pairs, found %s' % n)
    for stem, split, ph, sk in rows:
        for sub, src in (('a', ph), ('b', sk)):
            d = os.path.join(args.out, split, sub)
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(src, os.path.join(d, stem + os.path.splitext(src)[1].lower()))
    with open(os.path.join(args.out, 'ORIGINAL_NAMES.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stem', 'split', 'original_photo', 'original_sketch'])
        for stem, split, ph, sk in sorted(rows, key=lambda r: (r[1] != 'train', r[0])):
            w.writerow([stem, split, os.path.basename(ph), os.path.basename(sk)])
    print('cuhk: %d train / %d test pairs -> %s' % (n['train'], n['test'], args.out))


if __name__ == '__main__':
    main()
