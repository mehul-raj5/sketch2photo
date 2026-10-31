#!/usr/bin/env python3
"""build_cufsf.py — aligned sketch/photo pairs from CUFSF sketches (CUHK MMLab) and NIST FERET photos.

  --variant color  -> data/datasets/color_feret   Color FERET photos (853 of the 1,194 CUFSF subjects have one)
  --variant gray   -> data/datasets/cufsf         original gray FERET photos, all 1,194 subjects (the CUFSF benchmark)

Run from the project root with Pillow and NumPy available, for example:
  uv run --no-project --with pillow --with numpy python 0_data_preparation/build_cufsf.py --variant color --feret <colorferet root> --preview /tmp/prev
  uv run --no-project --with pillow --with numpy python 0_data_preparation/build_cufsf.py --variant color --feret <colorferet root> --apply

Pairing    sketch original_sketch/<id>.jpg + sketch_points/<id>.3pts  <->  photo_points/<FERET name>.3pts (one photo per
           subject; id = first 5 characters of the FERET name).
           gray photo   <feret>/dvd2/gray_feret_cd{1,2}/data/images/<FERET name>.tif.bz2 — 256x384; CUFSF photo points
                        are given for the photo resized 2x and padded 100 px left/right.
           colour photo <feret>/dvd{1,2}/<entry of doc/gray_to_color_feret.map>.bz2 — 512x768; points minus 100 px in x.
Alignment  similarity transform placing both eye centres (points 1 and 2) at fixed positions of the output frame,
           separately for sketch and photo. Default framing = CUHK's official CUFSF crops scaled to 200x250
           (eyes at (64.1, 110.9) and (138.7, 110.9)); looser framings push FERET close-ups far outside the photo. Photos PNG (RGB colour / L gray), sketches PNG (L).
Clean-up   before warping, each sketch scan loses its printed file-name label and thin scanner/paper-edge lines;
           photo areas beyond the FERET frame are filled by edge replication (no flat bands).
Split      DECP Table I uses 250 training pairs for CUFSF: the first 250 available subjects in ID order are train.
Check      every sketch is mapped into CUHK's official 64x80 crop (eyes measured at (20.50, 35.48) and (44.38, 35.48))
           and compared with cropped_sketch/<id>.jpg by normalised cross-correlation (NCC).
"""
import argparse
import bz2
import csv
import datetime
import hashlib
import io
import os
import sys
import zipfile

import numpy as np
from PIL import Image

OFFICIAL_CROP_EYES = ((20.50, 35.48), (44.38, 35.48))  # measured from cropped_sketch.zip, 64x80 frame


def read_pts(z, name):
    rows = [line.split() for line in z.read(name).decode("latin1").splitlines() if line.strip()]
    return [(float(x), float(y)) for x, y in rows[:3]]


def similarity(src_l, src_r, dst_l, dst_r):
    zl, zr, wl, wr = complex(*src_l), complex(*src_r), complex(*dst_l), complex(*dst_r)
    a = (wr - wl) / (zr - zl)
    return a, wl - a * zl


def align(img, eye_l, eye_r, dst_l, dst_r, size, fill, valid=None):
    """Warp img so its eyes land on dst_l/dst_r; returns (image, fraction of output pixels outside the valid source)."""
    valid = valid if valid is not None else Image.new("L", img.size, 255)
    a, _ = similarity(eye_l, eye_r, dst_l, dst_r)
    s = abs(a)
    if s < 0.95:  # shrink with LANCZOS first so the affine step doesn't alias
        w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
        fx, fy = w / img.width, h / img.height
        img, valid = img.resize((w, h), Image.LANCZOS), valid.resize((w, h), Image.NEAREST)
        eye_l, eye_r = (eye_l[0] * fx, eye_l[1] * fy), (eye_r[0] * fx, eye_r[1] * fy)
        a, _ = similarity(eye_l, eye_r, dst_l, dst_r)
    _, b = similarity(eye_l, eye_r, dst_l, dst_r)
    q = 1 / a
    c = -q * b
    data = (q.real, -q.imag, c.real, q.imag, q.real, c.imag)
    out = img.transform(size, Image.AFFINE, data=data, resample=Image.BICUBIC, fillcolor=fill)
    inside = valid.transform(size, Image.AFFINE, data=data, resample=Image.NEAREST, fillcolor=0)
    return out, float((np.asarray(inside) == 0).mean())


def pad_edge(img, eye_l, eye_r):
    """Replicate the border so warps never show flat bands; returns padded image, shifted eyes and validity mask."""
    a = np.asarray(img)
    pad = max(img.width, img.height) // 2
    widths = ((pad, pad), (pad, pad)) + (((0, 0),) if a.ndim == 3 else ())
    valid = np.zeros(a.shape[:2], dtype=np.uint8)
    valid = np.pad(valid + 255, ((pad, pad), (pad, pad)))
    return (Image.fromarray(np.pad(a, widths, mode="edge")), (eye_l[0] + pad, eye_l[1] + pad),
            (eye_r[0] + pad, eye_r[1] + pad), Image.fromarray(valid))


def clean_sketch(img, ink=128, min_dark=3, gap_rows=8, max_text_rows=40, split_rows=3):
    """Remove scan artefacts from a CUFSF sketch page before warping.

    1. Thin near-full-height lines (dark in > 80 % of rows) in the outer 15 % of the page width — scanner/paper
       edges — are whitened, and so are thin line segments (<= 8 px) crossing the top 15 % of the page.
    2. The printed file-name label above the head. Ink rows in the central 70 % of columns are grouped into runs
       (split by >= `split_rows` blank rows). The last run that starts in the top 10 % of the page, ends in the top
       13 %, is at most `max_text_rows` tall and is followed by `gap_rows` blank rows is the label; the page is
       whitened from the top down to it. A label touching the hair is left in place.
    Returns (image, last whitened label row or -1, number of whitened line columns).
    """
    a = np.asarray(img).copy()
    h, w = a.shape
    colfrac = (a < ink).mean(0)
    line_cols = [x for x in range(w) if colfrac[x] > 0.8 and (x < 0.15 * w or x > 0.85 * w)]
    for x in line_cols:
        a[:, max(0, x - 1): x + 2] = 255
    top = int(0.15 * h)  # thin line segments crossing the label band (<= 8 px wide, dark in > 60 % of its rows)
    groups = []
    for x in np.nonzero((a[:top] < ink).mean(0) > 0.6)[0]:
        if groups and x == groups[-1][1] + 1:
            groups[-1][1] = x
        else:
            groups.append([x, x])
    for s, e in groups:
        if e - s + 1 <= 8:
            a[:top, max(0, s - 1): e + 2] = 255
            line_cols.extend(range(s, e + 1))
    inked = (a[:, int(0.15 * w): int(0.85 * w)] < ink).sum(1) > min_dark
    runs, r = [], 0
    while r < int(0.13 * h):
        if not inked[r]:
            r += 1
            continue
        s = r
        while r < h and inked[r: r + split_rows].any():
            r += 1
        runs.append((s, r - 1))
    label_end = -1
    for s, e in runs:
        if s <= 0.10 * h and e <= 0.13 * h and e - s + 1 <= max_text_rows and not inked[e + 1: e + 1 + gap_rows].any():
            label_end = e + 2
    if label_end >= 0:
        a[: label_end + 1, :] = 255
    return Image.fromarray(a), label_end, len(line_cols)


def border_fill(img):  # kept for reference; photos now use pad_edge
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[..., None]
    med = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0), axis=0).astype(int)
    return int(med[0]) if img.mode == "L" else tuple(int(v) for v in med)


def ncc(u, v):
    u = u - u.mean()
    v = v - v.mean()
    return float((u * v).sum() / (np.sqrt((u * u).sum() * (v * v).sum()) + 1e-9))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["color", "gray"], required=True)
    ap.add_argument("--feret", required=True, help="NIST colorferet release root (contains dvd1/ and dvd2/)")
    ap.add_argument("--sources", default="data/sources/cufsf_cuhk")
    ap.add_argument("--datasets", default="data/datasets")
    ap.add_argument("--eyes", default="64.1,110.9,138.7,110.9",
                    help="left-eye x,y,right-eye x,y in the output frame (default: CUHK's official CUFSF crop framing, scaled to 200x250)")
    ap.add_argument("--size", default="200x250")
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--preview", help="dry run: write side-by-side previews (photo | sketch | blend) of 8 pairs here")
    ap.add_argument("--ids", help="with --preview: comma-separated subject ids to render instead of 8 spread-out ones")
    ap.add_argument("--apply", action="store_true", help="build the dataset")
    args = ap.parse_args()

    feret = os.path.expanduser(args.feret)
    W, H = map(int, args.size.split("x"))
    ex = [float(v) for v in args.eyes.split(",")]
    dst_l, dst_r = (ex[0], ex[1]), (ex[2], ex[3])
    name = "color_feret" if args.variant == "color" else "cufsf"
    out = os.path.join(args.datasets, name)
    S = args.sources

    zs, zp, zo, zc = (zipfile.ZipFile(os.path.join(S, f"{n}.zip")) for n in ("sketch_points", "photo_points", "original_sketch", "cropped_sketch"))
    listed = {os.path.splitext(l.strip())[0] for l in open(os.path.join(S, "feret_filenames.txt"), encoding="latin1") if l.strip()}
    photo_names = {os.path.splitext(os.path.basename(n))[0] for n in zp.namelist() if n.endswith(".3pts")}
    sketch_ids = sorted(os.path.splitext(os.path.basename(n))[0] for n in zs.namelist() if n.endswith(".3pts"))
    by_id = {n[:5]: n for n in photo_names}
    if not (len(by_id) == len(photo_names) == len(sketch_ids) == len(listed) == 1194 and set(by_id) == set(sketch_ids) and listed == photo_names):
        sys.exit("CUFSF source files are inconsistent — nothing built")

    src = {}
    if args.variant == "gray":
        index = {}
        for cd in ("gray_feret_cd1", "gray_feret_cd2"):
            d = os.path.join(feret, "dvd2", cd, "data", "images")
            for f in os.listdir(d):
                if f.endswith(".tif.bz2"):
                    index[f[: -len(".tif.bz2")]] = os.path.join(d, f)
        src = {i: index.get(by_id[i]) for i in sketch_ids}
    else:
        cmap = {}
        with open(os.path.join(feret, "dvd1", "doc", "gray_to_color_feret.map")) as fh:
            for line in fh:
                p = line.split()
                if len(p) == 2:
                    cmap[os.path.splitext(os.path.basename(p[0]))[0]] = p[1]
        for i in sketch_ids:
            src[i] = None
            rel = cmap.get(by_id[i])
            for dvd in ("dvd1", "dvd2"):
                if rel and os.path.exists(os.path.join(feret, dvd, rel + ".bz2")):
                    src[i] = os.path.join(feret, dvd, rel + ".bz2")
                    break

    avail = [i for i in sketch_ids if src[i]]
    train = set(avail[: args.n_train])
    print(f"{name}: {len(avail)} of {len(sketch_ids)} CUFSF subjects have a {args.variant} photo -> train {len(train)} / test {len(avail) - len(train)}")
    if args.variant == "gray" and len(avail) != len(sketch_ids):
        sys.exit("gray FERET photos missing — nothing built")
    if args.apply:
        existing = os.path.join(out, "train", "a")
        if os.path.isdir(existing) and os.listdir(existing):
            sys.exit(f"{out} already contains images — remove it first")
        todo = avail
    elif args.preview:
        wanted = [s.strip() for s in args.ids.split(",")] if args.ids else None
        todo = [i for i in avail if i in wanted] if wanted else avail[:: max(1, len(avail) // 8)][:8]
    else:
        print("dry run only: add --preview DIR to render sample pairs, or --apply to build")
        return

    rows, scores, oob_ph, oob_sk = [], [], [], []
    labels = 0
    for k, i in enumerate(todo):
        split = "train" if i in train else "test"
        sp = read_pts(zs, f"sketch_points/{i}.3pts")
        sketch = Image.open(io.BytesIO(zo.read(f"original_sketch/{i}.jpg"))).convert("L")
        sketch, label_end, n_line_cols = clean_sketch(sketch)
        labels += label_end >= 0
        sk_out, sk_oob = align(sketch, sp[0], sp[1], dst_l, dst_r, (W, H), 255)

        pp = read_pts(zp, f"photo_points/{by_id[i]}.3pts")
        with open(src[i], "rb") as fh:
            photo = Image.open(io.BytesIO(bz2.decompress(fh.read())))
        if args.variant == "gray":
            photo = photo.convert("L")
            pl, pr = ((pp[0][0] - 100) / 2, pp[0][1] / 2), ((pp[1][0] - 100) / 2, pp[1][1] / 2)
        else:
            photo = photo.convert("RGB")
            pl, pr = (pp[0][0] - 100, pp[0][1]), (pp[1][0] - 100, pp[1][1])
        padded, pl, pr, valid = pad_edge(photo, pl, pr)
        ph_out, ph_oob = align(padded, pl, pr, dst_l, dst_r, (W, H), 0, valid)

        check, _ = align(sk_out, dst_l, dst_r, OFFICIAL_CROP_EYES[0], OFFICIAL_CROP_EYES[1], (64, 80), 255)
        ref = np.asarray(Image.open(io.BytesIO(zc.read(f"cropped_sketch/{i}.jpg"))).convert("L"), dtype=np.float32)
        score = ncc(np.asarray(check, dtype=np.float32), ref)
        scores.append(score); oob_ph.append(ph_oob); oob_sk.append(sk_oob)

        if args.apply:
            for d in ("a", "b"):
                os.makedirs(os.path.join(out, split, d), exist_ok=True)
            ph_out.save(os.path.join(out, split, "a", f"{i}.png"))
            sk_out.save(os.path.join(out, split, "b", f"{i}.png"))
            rows.append([i, split, f"original_sketch/{i}.jpg", by_id[i], os.path.relpath(src[i], feret), sha256(src[i]),
                         f"{score:.4f}", f"{ph_oob:.4f}", f"{sk_oob:.4f}", label_end])
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(todo)}")
        else:
            os.makedirs(args.preview, exist_ok=True)
            canvas = Image.new("RGB", (W * 3, H), "white")
            canvas.paste(ph_out.convert("RGB"), (0, 0))
            canvas.paste(sk_out.convert("RGB"), (W, 0))
            canvas.paste(Image.blend(ph_out.convert("RGB"), sk_out.convert("RGB"), 0.5), (2 * W, 0))
            canvas.save(os.path.join(args.preview, f"{name}_{i}_{split}.png"))
            print(f"  preview {i} ({split}): NCC vs official crop {score:.3f} | outside source: photo {ph_oob:.1%}, sketch {sk_oob:.1%}")

    s = np.array(scores)
    low = [(r[0], r[6]) for r in rows if float(r[6]) < 0.6]
    summary = (f"NCC vs CUHK's official crops: median {np.median(s):.3f}, 5th percentile {np.percentile(s, 5):.3f}, min {s.min():.3f}; "
               f"pixels beyond the source (edge-replicated for photos): photo max {max(oob_ph):.1%}, sketch max {max(oob_sk):.1%}; "
               f"file-name label whitened on {labels} of {len(todo)} sketches")
    print(summary)
    if not args.apply:
        return

    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    with open(os.path.join(out, "meta", "ORIGINAL_NAMES.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "split", "cufsf_sketch", "feret_name", "photo_source", "photo_source_sha256", "ncc_vs_official_crop", "photo_outside_fraction", "sketch_outside_fraction", "label_whitened_to_row"])
        w.writerows(rows)
    src_hashes = "\n".join(f"| `{f}` | {sha256(os.path.join(S, f))[:12]} |" for f in sorted(os.listdir(S)) if not f.startswith("."))
    n_train = sum(r[1] == "train" for r in rows)
    with open(os.path.join(out, "meta", "PROCEDURE.md"), "w") as fh:
        fh.write(f"""# {name} — how this dataset was built

Built {datetime.date.today()} with `0_data_preparation/build_cufsf.py --variant {args.variant} --feret <colorferet> --apply` (eyes {args.eyes}, size {args.size}, {args.n_train} training subjects).

- **Sketches:** CUFSF artist sketches from CUHK MMLab (`http://mmlab.ie.cuhk.edu.hk/archive/cufsf/files/`), stored in `data/sources/cufsf_cuhk/`.
- **Photos:** NIST FERET release `colorferet` ({'Color FERET via doc/gray_to_color_feret.map' if args.variant == 'color' else 'original gray FERET, dvd2/gray_feret_cd1-2'}). FERET images are licensed by NIST — do not redistribute these files.
- **Pairs:** {len(rows)} (train {n_train} / test {len(rows) - n_train}); {len(sketch_ids) - len(avail)} of the 1,194 CUFSF subjects have no {args.variant} photo.
- **Alignment:** similarity transform putting both eye centres at {dst_l} and {dst_r} of a {W}x{H} frame (CUHK's official CUFSF crop framing, measured on cropped_sketch.zip and scaled from 64x80), computed separately from the CUFSF sketch points and photo points (photo points are defined for the 2x-resized gray image padded 100 px left and right).
- **Clean-up:** before warping, the printed file-name label above the head and thin scanner/paper-edge lines are whitened on each sketch scan; photo areas beyond the FERET frame are filled by edge replication.
- **Split:** the first {args.n_train} available subjects in ID order are train, the rest test (DECP Table I uses 250 training pairs for CUFSF).
- **Check:** {summary}. Pairs below NCC 0.6: {low if low else 'none'}.
- **Rename map:** `ORIGINAL_NAMES.csv` (stem = 5-digit FERET subject id).

| CUFSF source file | sha256 (12) |
|---|---|
{src_hashes}
""")
    keep = os.path.join(out, ".gitkeep")
    if os.path.exists(keep):
        os.remove(keep)
    ok = True
    for split in ("train", "test"):
        a = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, split, "a"))}
        b = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, split, "b"))}
        ok &= a == b
        print(f"  {split}: photos {len(a)} sketches {len(b)} paired {'yes' if a == b else 'NO'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
