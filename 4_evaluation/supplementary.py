"""SSIM and PSNR (T9, supplementary only — never the headline): skimage on the prepared 8-bit RGB images."""
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def ssim_psnr(pred_path, gt_path):
    a = np.asarray(Image.open(pred_path).convert('RGB'))
    b = np.asarray(Image.open(gt_path).convert('RGB'))
    ssim = structural_similarity(b, a, channel_axis=2, data_range=255)
    psnr = peak_signal_noise_ratio(b, a, data_range=255)
    return float(ssim), float(psnr)
