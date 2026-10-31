"""FSIM: NumPy port of Zhang et al.'s FeatureSIM.m (IEEE TIP 2011) with its embedded phasecong2 / lowpassfilter,
driven like DECP's compute_fsim.m: imread -> rgb2gray (uint8, rounded) -> double in [0, 255] -> FSIM.

Checked against GNU Octave running the original FeatureSIM.m (reference/FeatureSIM.m).
Only the first output (FSIM) is used; for 2-D inputs FeatureSIM's FSIMc equals FSIM.
"""
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

import protocol as P

NSCALE = 4
NORIENT = 4
MIN_WAVELENGTH = 6
MULT = 2
SIGMA_ONF = 0.55
DTHETA_ON_SIGMA = 1.2
K = 2.0
EPSILON = 1e-4
T1 = 0.85
T2 = 160.0
DX = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64) / 16.0
DY = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float64) / 16.0


def matlab_round(x):
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def rgb2gray_uint8(rgb):
    """MATLAB rgb2gray on uint8 RGB: weighted sum, rounded, saturated to uint8."""
    rgb = np.asarray(rgb, dtype=np.float64)
    c = P.RGB2GRAY
    y = c[0] * rgb[..., 0] + c[1] * rgb[..., 1] + c[2] * rgb[..., 2]
    return np.clip(matlab_round(y), 0, 255).astype(np.uint8)


def load_gray(path):
    """compute_fsim.m: imread; rgb2gray if 3 channels; double in [0, 255]."""
    im = Image.open(path)
    if im.mode not in ('L', 'RGB'):
        im = im.convert('RGB')
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = rgb2gray_uint8(arr)
    return arr.astype(np.float64)


def conv2_same(x, k):
    """MATLAB conv2(x, k, 'same'): central part of the full convolution, offset floor(size(k)/2), zero padding."""
    full = convolve2d(x, k, mode='full')
    r0, c0 = k.shape[0] // 2, k.shape[1] // 2
    return full[r0:r0 + x.shape[0], c0:c0 + x.shape[1]]


def _axis_range(n):
    if n % 2:
        return np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) / (n - 1)
    return np.arange(-n / 2.0, n / 2.0) / n


def _lowpassfilter(rows, cols, cutoff, n):
    x, y = np.meshgrid(_axis_range(cols), _axis_range(rows))
    radius = np.sqrt(x ** 2 + y ** 2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


_FILTER_CACHE = {}


def _filters(rows, cols):
    """Log-Gabor x angular filters and the per-orientation noise constants; depend on the image size only."""
    key = (rows, cols)
    if key in _FILTER_CACHE:
        return _FILTER_CACHE[key]
    theta_sigma = np.pi / NORIENT / DTHETA_ON_SIGMA
    x, y = np.meshgrid(_axis_range(cols), _axis_range(rows))
    radius = np.fft.ifftshift(np.sqrt(x ** 2 + y ** 2))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0
    sintheta, costheta = np.sin(theta), np.cos(theta)
    lp = _lowpassfilter(rows, cols, 0.45, 15)
    log_gabor = []
    for s in range(NSCALE):
        fo = 1.0 / (MIN_WAVELENGTH * MULT ** s)
        g = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(SIGMA_ONF) ** 2)) * lp
        g[0, 0] = 0.0
        log_gabor.append(g)
    per_orient = []
    for o in range(NORIENT):
        angl = o * np.pi / NORIENT
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-dtheta ** 2 / (2 * theta_sigma ** 2))
        filters = [log_gabor[s] * spread for s in range(NSCALE)]
        ifft_filt = [np.real(np.fft.ifft2(f)) * np.sqrt(rows * cols) for f in filters]
        em_n = np.sum(filters[0] ** 2)
        sum_an2 = np.sum(sum(f ** 2 for f in ifft_filt))
        sum_aiaj = 0.0
        for si in range(NSCALE - 1):
            for sj in range(si + 1, NSCALE):
                sum_aiaj += np.sum(ifft_filt[si] * ifft_filt[sj])
        per_orient.append((filters, em_n, sum_an2, sum_aiaj))
    _FILTER_CACHE[key] = per_orient
    return per_orient


def phasecong2(im):
    rows, cols = im.shape
    imagefft = np.fft.fft2(im)
    energy_all = np.zeros((rows, cols))
    an_all = np.zeros((rows, cols))
    for filters, em_n, sum_an2, sum_aiaj in _filters(rows, cols):
        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_an = np.zeros((rows, cols))
        eo = []
        for f in filters:
            r = np.fft.ifft2(imagefft * f)
            eo.append(r)
            sum_an += np.abs(r)
            sum_e += r.real
            sum_o += r.imag
        x_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + EPSILON
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy
        energy = np.zeros((rows, cols))
        for r in eo:
            e, o = r.real, r.imag
            energy += e * mean_e + o * mean_o - np.abs(e * mean_o - o * mean_e)
        median_e2n = np.median(np.abs(eo[0]).ravel() ** 2)
        mean_e2n = -median_e2n / np.log(0.5)
        noise_power = mean_e2n / em_n
        est_noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_aiaj
        tau = np.sqrt(est_noise_energy2 / 2)
        est_noise_energy = tau * np.sqrt(np.pi / 2)
        est_noise_energy_sigma = np.sqrt((2 - np.pi / 2) * tau ** 2)
        t = (est_noise_energy + K * est_noise_energy_sigma) / 1.7
        energy = np.maximum(energy - t, 0.0)
        energy_all += energy
        an_all += sum_an
    return energy_all / an_all


def feature_sim(y1, y2):
    """FeatureSIM(imageRef, imageDis) for 2-D double images in [0, 255]; returns FSIM."""
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    if y1.shape != y2.shape or y1.ndim != 2:
        raise ValueError('FSIM needs two 2-D images of the same size, got %s and %s' % (y1.shape, y2.shape))
    rows, cols = y1.shape
    f = int(max(1, matlab_round(min(rows, cols) / 256.0)))
    if f > 1:
        k = np.ones((f, f)) / (f * f)
        y1 = conv2_same(y1, k)[0:rows:f, 0:cols:f]
        y2 = conv2_same(y2, k)[0:rows:f, 0:cols:f]
    pc1 = phasecong2(y1)
    pc2 = phasecong2(y2)
    g1 = np.sqrt(conv2_same(y1, DX) ** 2 + conv2_same(y1, DY) ** 2)
    g2 = np.sqrt(conv2_same(y2, DX) ** 2 + conv2_same(y2, DY) ** 2)
    pc_sim = (2 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
    g_sim = (2 * g1 * g2 + T2) / (g1 ** 2 + g2 ** 2 + T2)
    pcm = np.maximum(pc1, pc2)
    return float(np.sum(g_sim * pc_sim * pcm) / np.sum(pcm))


def fsim_files(pred_path, gt_path):
    """compute_fsim.m order: fsim(probe = prediction, gallery = GT); FSIM is symmetric anyway."""
    return feature_sim(load_gray(pred_path), load_gray(gt_path))
