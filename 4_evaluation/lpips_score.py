"""LPIPS v0.1 with alex, squeeze and vgg, exactly as 4_evaluation/decp/evaluate/lpips_fid_2dirs.py.

pip lpips 0.1.4 = DECP's bundled copy (identical v0.1 head weights by sha256; load_image / im2tensor identical):
cv2 decode -> RGB -> [-1, 1], no resize, per-image distance, mean over images.
"""
import lpips
import torch

import protocol as P


class LPIPSScorer(object):
    def __init__(self, device):
        self.device = device
        self.nets = {n: lpips.LPIPS(net=n, version='0.1', verbose=False).to(device).eval() for n in P.LPIPS_NETS}

    @torch.no_grad()
    def score(self, pred_path, gt_path):
        img0 = lpips.im2tensor(lpips.load_image(pred_path)).to(self.device)
        img1 = lpips.im2tensor(lpips.load_image(gt_path)).to(self.device)
        return {n: float(net.forward(img0, img1).item()) for n, net in self.nets.items()}
