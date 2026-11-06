"""Image metrics: PSNR, SSIM, relative L2."""
import math

import numpy as np
import torch

from .utils import pip_install


def psnr(x, gt, data_range=1.0):
    mse = torch.mean((x - gt) ** 2).item()
    return 10 * math.log10(data_range ** 2 / max(mse, 1e-12))


def ssim(x, gt, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        pip_install("scikit-image")
        from skimage.metrics import structural_similarity
    return float(structural_similarity(x.detach().cpu().numpy().astype(np.float64),
                                       gt.detach().cpu().numpy().astype(np.float64), data_range=data_range))


def rel_l2(x, gt):
    return (torch.linalg.norm(x - gt) / torch.linalg.norm(gt).clamp(min=1e-12)).item()
