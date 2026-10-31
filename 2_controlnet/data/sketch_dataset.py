"""Sketch-only dataset for generation (P2, P4 J1-J2, D6, NEW-pipeline-3).

Opens only `<dataset>/<train|test>/b`. There is deliberately no code path to the photo folder `a/`,
no prompt file and no reference image: generation sees the sketch and nothing else.
- split 'test'            : every sketch stem in test/b (final runs only, D5)
- split 'val' / 'train_dev': stems from data/splits/<id>/<split>.txt, sketches from train/b
Geometry is identical to training: PIL bicubic squash of the original to 512² (edge path) and 256² (codebook).
"""
import os

from torch.utils.data import Dataset

from data.image_io import index_images, load_rgb, natural_key, pil_to_tensor, read_stem_list, squash

GENERATION_SPLITS = ('val', 'train_dev', 'test')


class SketchOnlyDataset(Dataset):
    def __init__(self, dataset_dir, splits_dir, split, edge_size=512, sketch_size=256, limit=None, prompts=None):
        if split not in GENERATION_SPLITS:
            raise ValueError('split must be one of %s, got %r' % (GENERATION_SPLITS, split))
        folder = 'test' if split == 'test' else 'train'
        self.sketch_dir = os.path.join(dataset_dir, folder, 'b')
        sketches = index_images(self.sketch_dir)
        if split == 'test':
            stems = sorted(sketches, key=natural_key)
        else:
            stems = read_stem_list(os.path.join(splits_dir, split + '.txt'))
        missing = [s for s in stems if s not in sketches]
        if missing:
            raise ValueError('%d stems of split %s have no sketch in %s (first: %s)'
                             % (len(missing), split, self.sketch_dir, missing[:5]))
        self.complete = limit is None or limit >= len(stems)
        if limit is not None:
            stems = stems[:int(limit)]
        self.split = split
        self.stems = list(stems)
        self.paths = [sketches[s] for s in self.stems]
        self.edge_size = int(edge_size)
        self.sketch_size = int(sketch_size)
        self.prompts = prompts   # attribute prompts (oracle) or None -> the fixed text is used by the caller

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        sketch = load_rgb(self.paths[index])
        return {
            'stem': self.stems[index],
            'prompt': self.prompts.text(self.stems[index]) if self.prompts is not None else '',
            'sketch': pil_to_tensor(squash(sketch, self.sketch_size)),     # (3, 256, 256) [-1, 1] -> CVQ-VAE
            'sketch_raw': pil_to_tensor(squash(sketch, self.edge_size)),  # (3, 512, 512) [-1, 1] -> edge path
        }
