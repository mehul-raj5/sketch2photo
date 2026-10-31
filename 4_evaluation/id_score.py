"""ID score as 4_evaluation/decp/evaluate/calc_id_loss_parallel.py: DECP's MTCNN `align` (thresholds 0.15/0.25/0.35, first
detection, 112 x 112 similarity warp) on prediction and GT, CurricularFace IR-101 (L2-normalised 512-d), score =
dot product. A pair is skipped when either image has no detection; skips are counted and listed.

DECP evaluate package is imported read-only: its cwd-relative MTCNN weight paths and hard-coded `device = 'cuda:0'` globals are
overridden in memory (same approach as 2_controlnet/tools/precompute_landmarks.py).
"""
import hashlib
import os
import sys

import torch
import torchvision.transforms as trans
from PIL import Image

import protocol as P


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _import_harness(device):
    evaluate_dir = os.path.join(P.DECP_DIR, 'evaluate')
    if 'models' in sys.modules:
        raise RuntimeError('a package named "models" is already imported — it would shadow 4_evaluation/decp/evaluate/models')
    weights = {}
    for net in ('pnet', 'rnet', 'onet'):
        path = os.path.join(evaluate_dir, 'mtcnn', net + '.npy')
        if not os.path.isfile(path):
            path = os.path.join(evaluate_dir, 'models', 'mtcnn', 'mtcnn_pytorch', 'src', 'weights', net + '.npy')
        weights[net] = path
    for path in (P.DECP_DIR, evaluate_dir):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, P.DECP_DIR)
    sys.path.insert(0, evaluate_dir)
    import evaluate.paths_config as paths_config
    for net, path in weights.items():
        paths_config.model_paths['mtcnn_' + net] = path
    import models.mtcnn.mtcnn_pytorch.src.first_stage as first_stage
    import models.mtcnn.mtcnn as harness_mtcnn
    from models.encoders.model_irse import IR_101
    first_stage.device = device
    harness_mtcnn.device = device
    if not os.path.abspath(harness_mtcnn.__file__).startswith(evaluate_dir):
        raise RuntimeError('MTCNN imported from %s, expected 4_evaluation/decp/evaluate' % harness_mtcnn.__file__)
    return harness_mtcnn, IR_101, {k: _sha256(v) for k, v in weights.items()}


class IDScorer(object):
    def __init__(self, device):
        self.device = device
        mtcnn_mod, IR_101, self.mtcnn_sha256 = _import_harness(str(device))
        self.mtcnn = mtcnn_mod.MTCNN()
        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(P.CURRICULARFACE, map_location='cpu'))
        self.facenet.to(device).eval()
        self.transform = trans.Compose([trans.ToTensor(), trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.weights_sha256 = _sha256(P.CURRICULARFACE)

    def _align(self, path):
        im = Image.open(path)
        if im.mode == 'L':
            im = im.convert('RGB')
        try:
            face, _ = self.mtcnn.align(im)
        except ValueError:  # empty P-Net stage: np.vstack([]) inside detect_faces — DECP's script would crash
            return None, 'error'
        return face, (None if face is not None else 'none')

    @torch.no_grad()
    def _embed(self, face):
        return self.facenet(self.transform(face).unsqueeze(0).to(self.device))[0]

    def score(self, pred_path, gt_path):
        """Returns (score or None, skip reason '' / 'pred' / 'gt'). DECP aligns the prediction first."""
        pred_face, why = self._align(pred_path)
        if pred_face is None:
            return None, 'pred' + ('_error' if why == 'error' else '')
        gt_face, why = self._align(gt_path)
        if gt_face is None:
            return None, 'gt' + ('_error' if why == 'error' else '')
        return float(self._embed(pred_face).dot(self._embed(gt_face))), ''
