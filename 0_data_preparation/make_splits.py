#!/usr/bin/env python3
"""Write a tuning split from a dataset's training pairs (PLAN D5, P0.8).

Default: WildSketch, 40 of the 396 training stems -> val.txt, the rest -> train_dev.txt, seed 0.
Deterministic: stems in natural order, random.Random(seed).sample. Refuses to overwrite unless --force.
Run from anywhere:  python3 0_data_preparation/make_splits.py --dataset wildsketch --n-val 40 --seed 0
"""
import argparse
import os
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="wildsketch")
ap.add_argument("--n-val", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")   # <repo>/data
train = os.path.join(root, "datasets", args.dataset, "train")
natural = lambda s: (0, int(s), s) if s.isdigit() else (1, 0, s)


def stems(sub):
    return sorted((os.path.splitext(f)[0] for f in os.listdir(os.path.join(train, sub)) if not f.startswith(".")), key=natural)


photos, sketches = stems("a"), stems("b")
if photos != sketches:
    sys.exit(f"train/a and train/b stems differ ({len(photos)} vs {len(sketches)})")
val = sorted(random.Random(args.seed).sample(photos, args.n_val), key=natural)
chosen = set(val)
dev = [s for s in photos if s not in chosen]

out = os.path.join(root, "splits", args.dataset)
os.makedirs(out, exist_ok=True)
paths = {name: os.path.join(out, name + ".txt") for name in ("train_dev", "val")}
if not args.force and any(os.path.exists(p) for p in paths.values()):
    sys.exit(f"split files already exist in {out}; use --force to overwrite")
for name, items in (("train_dev", dev), ("val", val)):
    with open(paths[name], "w") as f:
        f.write("\n".join(items) + "\n")
print(f"{args.dataset}: train_dev {len(dev)} · val {len(val)} (seed {args.seed}) -> {out}")
