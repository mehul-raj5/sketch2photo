"""5-point GT landmarks for the aligned ID loss (P7, M5) — run once on the server, training photos only.

    python 2_controlnet/tools/precompute_landmarks.py --config 2_controlnet/configs/wildsketch.yaml --device cuda:0

- Detector: the evaluation harness's own MTCNN (4_evaluation/decp/evaluate/models/mtcnn, weights .npy in the repo — no
  download). DECP evaluate package is used read-only: its cwd-relative weight paths (evaluate/paths_config.py) and its hard-coded
  `device = 'cuda:0'` globals (mtcnn.py, first_stage.py) are overridden in memory; no file there is edited.
- Image: every paired training GT photo (train/a; never test/, D6) squashed with PIL bicubic to 256² — exactly the
  image the harness aligns — and landmarks[0], as MTCNN.align takes it. A photo without a detection is listed in
  `missing`; the training loss then uses DECP's fixed IDLoss crop for it (data/sd_dataset.py).
- Output (paths.landmarks, default <outputs>/wildsketch/landmarks/mtcnn5_train.json, written atomically): landmarks
  in the squash-512² frame, x512 = 2·x256 + 0.5 (pixel centres at integer coordinates), the raw 256² points, boxes,
  face counts, weight hashes and M5 sanity statistics (detection rate, inter-ocular distance, roll).
"""
import argparse
import json
import math
import os
import sys
import time

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from utils import config as C  # noqa: E402  (no torch import)

POINT_NAMES = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
DETECT_SIZE = 256
FRAME_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description='MTCNN 5-point landmarks of the training GT photos (P7)')
    p.add_argument('--config', required=True, help='v2 YAML config (dataset, paths.landmarks)')
    p.add_argument('--device', required=True, help='torch device for MTCNN, e.g. cuda:0 — never defaulted (D12)')
    p.add_argument('--out', default=None, help='output JSON (default: paths.landmarks of the config)')
    p.add_argument('--overwrite', action='store_true', help='replace an existing landmark file')
    return p.parse_args()


def import_harness_mtcnn(project_root, device):
    """Import 4_evaluation/decp/evaluate/models/mtcnn with absolute weight paths and our device, without editing DECP."""
    decp = os.path.join(project_root, '4_evaluation', 'decp')
    evaluate_dir = os.path.join(decp, 'evaluate')
    if 'models' in sys.modules:
        raise RuntimeError('a package named "models" is already imported — it would shadow the harness MTCNN')
    search = [os.path.join(evaluate_dir, 'mtcnn'),
              os.path.join(evaluate_dir, 'models', 'mtcnn', 'mtcnn_pytorch', 'src', 'weights')]
    weights = {}
    for net in ('pnet', 'rnet', 'onet'):
        found = [os.path.join(d, net + '.npy') for d in search if os.path.isfile(os.path.join(d, net + '.npy'))]
        if not found:
            raise FileNotFoundError('MTCNN %s weights not found in %s' % (net, search))
        weights[net] = {'path': found[0], 'sha256': C.sha256_file(found[0])}
    for path in (decp, evaluate_dir):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, decp)           # evaluate.paths_config
    sys.path.insert(0, evaluate_dir)   # models.mtcnn...
    import evaluate.paths_config as paths_config
    for net in ('pnet', 'rnet', 'onet'):
        paths_config.model_paths['mtcnn_' + net] = weights[net]['path']   # read by get_nets at import
    import models.mtcnn.mtcnn_pytorch.src.first_stage as first_stage
    import models.mtcnn.mtcnn as harness_mtcnn
    first_stage.device = device
    harness_mtcnn.device = device
    source = os.path.abspath(harness_mtcnn.__file__)
    if not source.startswith(os.path.abspath(evaluate_dir)):
        raise RuntimeError('imported MTCNN from %s, expected 4_evaluation/decp/evaluate' % source)
    return harness_mtcnn, weights, source


def _stats(values):
    if not values:
        return None
    s = sorted(values)
    return {'min': s[0], 'median': s[len(s) // 2], 'max': s[-1], 'n': len(s)}


def main():
    args = parse_args()
    cfg = C.load_config(args.config)
    C.apply_env(cfg)
    r = cfg['_resolved']
    out_path = os.path.abspath(args.out) if args.out else r['landmarks']
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit('%s exists — pass --overwrite to replace it' % out_path)

    # Our own modules first, so the names `data` / `utils` are bound to 2_controlnet; `models` is never imported here.
    from data.image_io import index_images, load_rgb, squash
    from data.sd_dataset import training_stems
    stems = training_stems(r['dataset_dir'], r['splits_dir'], 'all')
    photo_dir = os.path.join(r['dataset_dir'], 'train', 'a')
    photos = index_images(photo_dir)

    import numpy as np
    harness_mtcnn, weights, source = import_harness_mtcnn(r['project_root'], args.device)
    detector = harness_mtcnn.MTCNN()

    landmarks, landmarks_256, boxes_256, faces, missing, errors = {}, {}, {}, {}, [], {}
    eye_dist, roll = [], []
    t0 = time.time()
    for i, stem in enumerate(stems):
        image = squash(load_rgb(photos[stem]), DETECT_SIZE)
        try:
            boxes, points = detector.detect_faces(image)
        except ValueError as exc:  # the harness copy raises on an empty P-Net stage
            boxes, points = [], []
            errors[stem] = repr(exc)
        faces[stem] = int(len(points))
        if len(points) == 0:
            missing.append(stem)
            continue
        lm = np.asarray(points[0], dtype=np.float64)
        pts = [[float(lm[j]), float(lm[j + 5])] for j in range(5)]
        landmarks_256[stem] = pts
        landmarks[stem] = [[2.0 * x + 0.5, 2.0 * y + 0.5] for x, y in pts]
        boxes_256[stem] = [float(v) for v in np.asarray(boxes[0]).reshape(-1)[:5]]
        (lx, ly), (rx, ry) = pts[0], pts[1]
        eye_dist.append(math.hypot(rx - lx, ry - ly))
        roll.append(math.degrees(math.atan2(ry - ly, rx - lx)))
        if (i + 1) % 50 == 0:
            print('%d / %d photos' % (i + 1, len(stems)), flush=True)

    summary = {'photos': len(stems), 'detected': len(landmarks), 'missing': len(missing),
               'detection_rate': len(landmarks) / float(max(1, len(stems))),
               'multiple_faces': sum(1 for n in faces.values() if n > 1),
               'inter_ocular_px_256': _stats(eye_dist), 'roll_deg': _stats(roll),
               'abs_roll_over_15deg': sum(1 for a in roll if abs(a) > 15.0), 'seconds': time.time() - t0}
    payload = {
        'dataset': cfg['run']['dataset'], 'split': 'train, every paired stem (never test)',
        'frame': [FRAME_SIZE, FRAME_SIZE], 'points': POINT_NAMES,
        'convention': 'pixel centres at integer coordinates (MTCNN / cv2); detected on the PIL-bicubic %d² squash of '
                      'train/a; x%d = 2·x%d + 0.5' % (DETECT_SIZE, FRAME_SIZE, DETECT_SIZE),
        'landmarks': landmarks, 'landmarks_256': landmarks_256, 'boxes_256': boxes_256, 'faces_detected': faces,
        'missing': missing, 'errors': errors, 'summary': summary,
        'detector': {'source': source, 'call': 'MTCNN().detect_faces(image) defaults; landmarks[0] as MTCNN.align',
                     'weights': weights, 'device': args.device},
        'photos_dir': photo_dir, 'tool_sha256': C.sha256_file(os.path.abspath(__file__)),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    print(json.dumps(summary, indent=1))
    print('wrote %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
