#!/usr/bin/env python3
"""Build data/datasets/fs2k (official FS2K split: 1 058 train / 1 046 test pairs) from the FS2K release.

    python 0_data_preparation/prepare_fs2k.py --fs2k <FS2K root containing photo/, sketch/, anno_train.json, anno_test.json>

- stems: annotation `image_name` "photo<k>/image<NNNN>" -> "style<k>_<NNNN>";
  photo = photo/photo<k>/image<NNNN>.*, sketch = sketch/sketch<k>/sketch<NNNN>.* (style-2 sketches are .png);
- split from anno_train.json / anno_test.json;
- files copied byte for byte to fs2k/{train,test}/{a,b}/<stem>.<ext> (a = photo, b = sketch); the annotations go to fs2k/meta/.
"""
import argparse
import csv
import json
import os
import shutil
import sys


def find(folder, base):
    for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG'):
        p = os.path.join(folder, base + ext)
        if os.path.isfile(p):
            return p
    return None


def name_to_stem(image_name):
    """'photo1/image0110' -> ('style1_0110', 'photo1', 'image0110', 'sketch1', 'sketch0110')."""
    folder, base = image_name.split('/')
    k, num = folder.replace('photo', ''), base.replace('image', '')
    return 'style%s_%s' % (k, num), folder, base, 'sketch' + k, 'sketch' + num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fs2k', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datasets', 'fs2k'))
    args = ap.parse_args()
    rows = []
    for split in ('train', 'test'):
        for e in json.load(open(os.path.join(args.fs2k, 'anno_%s.json' % split))):
            stem, pf, pb, sf, sb = name_to_stem(e['image_name'])
            ph, sk = find(os.path.join(args.fs2k, 'photo', pf), pb), find(os.path.join(args.fs2k, 'sketch', sf), sb)
            if not ph or not sk:
                sys.exit('missing photo or sketch for %s' % e['image_name'])
            rows.append((stem, split, ph, sk))
    n = {s: sum(r[1] == s for r in rows) for s in ('train', 'test')}
    if n != {'train': 1058, 'test': 1046}:
        sys.exit('expected 1058 train / 1046 test pairs, found %s' % n)
    for stem, split, ph, sk in rows:
        for sub, src in (('a', ph), ('b', sk)):
            d = os.path.join(args.out, split, sub)
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(src, os.path.join(d, stem + os.path.splitext(src)[1].lower()))
    meta = os.path.join(args.out, 'meta')
    os.makedirs(meta, exist_ok=True)
    for f in ('anno_train.json', 'anno_test.json'):
        shutil.copyfile(os.path.join(args.fs2k, f), os.path.join(meta, f))
    with open(os.path.join(meta, 'ORIGINAL_NAMES.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stem', 'split', 'original_photo', 'original_sketch'])
        for stem, split, ph, sk in rows:
            w.writerow([stem, split, os.path.relpath(ph, args.fs2k), os.path.relpath(sk, args.fs2k)])
    print('fs2k: %d train / %d test pairs -> %s' % (n['train'], n['test'], args.out))


if __name__ == '__main__':
    main()
