"""Protocol constants of the evaluation harness (4_evaluation/CLAUDE.md, README.md §7-§8).

Changing anything that moves a score requires a new PROTOCOL version and a re-score of every run in RUNS.md.
"""
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
DECP_DIR = os.path.join(EVAL_DIR, 'decp')   # DECP's evaluation package (MTCNN alignment + CurricularFace backbone for ID)

# eval-v1 frozen 2026-09-22 after Gate G1 (user decision; FINDINGS F44). Hashes of every scoring file: RUNS.md.
PROTOCOL = 'eval-v1'

# Scoring size (width, height) per dataset — D15. Squash with PIL bicubic, aspect ratio not kept.
SCORING_SIZE = {
    'wildsketch': (256, 256),
    'fs2k': (256, 256),
    'cuhk': (200, 250),
    'color_feret': (200, 250),
    'cufsf': (200, 250),
}

# LPIPS network reported as the headline (the one that matches DECP's reported LPIPS).
HEADLINE_LPIPS = 'vgg'

# ID compares the prediction (as prepared at the scoring size) with the GT photo at its native size, as DECP's evaluation
# does. No effect where the native size already is the scoring size (cuhk, color_feret, cufsf).
ID_GT_NATIVE = True
LPIPS_NETS = ('alex', 'squeeze', 'vgg')

# FID exactly as DECP's recovered calculate.py: pytorch_fid, batch 32, pool3 2048-d.
FID_BATCH = 32
FID_DIMS = 2048

# KID: polynomial kernel (x.y/d + 1)^3, unbiased MMD^2; subsets as in clean-fid / torch-fidelity.
KID_SUBSETS = 100
KID_SUBSET_SIZE = 1000
KID_SEED = 0

# MATLAB rgb2gray coefficients (first row of inv(YIQ)); uint8 result rounded half away from zero.
RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)

CURRICULARFACE = os.path.join(DATA_ROOT, 'pretrained', 'eval', 'CurricularFace_Backbone.pth')
TORCH_CACHE = os.path.join(DATA_ROOT, 'pretrained', 'torch_cache')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
