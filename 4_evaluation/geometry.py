"""Pairing by stem and preparation of GT / prediction PNGs at the scoring size (README §8, D15)."""
import os

from PIL import Image

import protocol as P


class PairingError(RuntimeError):
    pass


def index_by_stem(folder):
    """{stem: path} for every image file in folder (other files, e.g. run.json, are ignored)."""
    if not os.path.isdir(folder):
        raise PairingError('not a folder: %s' % folder)
    index = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in P.IMAGE_EXTENSIONS:
            continue
        if stem in index:
            raise PairingError('duplicate stem %r in %s' % (stem, folder))
        index[stem] = os.path.join(folder, name)
    return index


def read_stem_list(path):
    with open(path) as fh:
        stems = [line.strip() for line in fh if line.strip()]
    if len(stems) != len(set(stems)):
        raise PairingError('duplicate stems in %s' % path)
    return stems


def gt_index(dataset, split):
    """GT photos of a split. val / train_dev are subsets of train/a given by data/splits/<id>/<split>.txt."""
    root = os.path.join(P.DATA_ROOT, 'datasets', dataset)
    if split in ('test', 'train'):
        return index_by_stem(os.path.join(root, split, 'a'))
    if split in ('val', 'train_dev'):
        full = index_by_stem(os.path.join(root, 'train', 'a'))
        stems = read_stem_list(os.path.join(P.DATA_ROOT, 'splits', dataset, split + '.txt'))
        missing = [s for s in stems if s not in full]
        if missing:
            raise PairingError('%d %s stems not in train/a, e.g. %s' % (len(missing), split, missing[:5]))
        return {s: full[s] for s in stems}
    raise PairingError('unknown split %r' % split)


def pair(gt, pred):
    """Exactly one prediction per GT stem; missing or extra stems are an error (README §8)."""
    missing = sorted(set(gt) - set(pred))
    extra = sorted(set(pred) - set(gt))
    if missing or extra:
        raise PairingError('prediction folder does not match the GT split: %d missing (e.g. %s), %d extra (e.g. %s)'
                           % (len(missing), missing[:5], len(extra), extra[:5]))
    return sorted(gt)


def prepare(src, dst, size):
    """RGB, PIL-bicubic squash to size (W, H) if needed, 8-bit PNG. Returns True if a resize happened."""
    im = Image.open(src).convert('RGB')
    resized = im.size != tuple(size)
    if resized:
        im = im.resize(tuple(size), Image.BICUBIC)
    im.save(dst, format='PNG')
    return resized


def prepare_folder(index, stems, out_dir, size):
    os.makedirs(out_dir, exist_ok=True)
    resized = []
    for s in stems:
        if prepare(index[s], os.path.join(out_dir, s + '.png'), size):
            resized.append(s)
    return resized
