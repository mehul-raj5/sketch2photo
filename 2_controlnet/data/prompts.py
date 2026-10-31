"""Prompt text per stem: one fixed sentence, or the attribute prompt of `tools/extract_attributes.py`.

The attribute files are derived from the ground-truth photos (`data/attributes/<dataset>/<split>.json`), so a
val or test run with `prompt.attributes.enabled: true` uses information no baseline has — it is the
"Ours + attributes" (witness-description) setting, reported as its own row (D6 amended 2026-09-22, F10).
`val` and `train_dev` read the train file, because those stems live in train/.
"""
import json
import os

SPLIT_FILE = {'train': 'train', 'train_dev': 'train', 'val': 'train', 'all': 'train', 'test': 'test'}


class Prompts(object):
    def __init__(self, cfg, split, resolved=None):
        p = cfg['prompt']
        self.fixed = p['text']
        self.enabled = bool(p['attributes']['enabled'])
        self.on_missing = p['attributes']['on_missing']
        self.split = split
        self.path = None
        self.table = {}
        if not self.enabled:
            return
        resolved = resolved or cfg.get('_resolved') or {}
        directory = resolved.get('attributes_dir')
        if directory is None:
            raise ValueError('prompt.attributes.enabled but the config has no resolved attributes_dir')
        name = SPLIT_FILE.get(split)
        if name is None:
            raise ValueError('no attribute file defined for split %r' % split)
        self.path = os.path.join(directory, name + '.json')
        if not os.path.isfile(self.path):
            raise FileNotFoundError('attribute file %s not found — run tools/extract_attributes.py '
                                    '(envs/deepface) for dataset split %r' % (self.path, name))
        with open(self.path) as fh:
            payload = json.load(fh)
        self.table = {k: v['prompt'] for k, v in payload['attributes'].items() if v.get('prompt')}
        self.source = payload.get('source')

    def text(self, stem):
        if not self.enabled:
            return self.fixed
        prompt = self.table.get(stem)
        if prompt is None:
            if self.on_missing == 'raise':
                raise KeyError('no attribute prompt for stem %r in %s' % (stem, self.path))
            return self.fixed
        return prompt

    def describe(self):
        return {'enabled': self.enabled, 'fixed_text': self.fixed, 'file': self.path,
                'stems_with_attributes': len(self.table), 'split': self.split,
                'source': getattr(self, 'source', None)}
