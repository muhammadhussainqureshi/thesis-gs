import math

import torch

from thesis_gs.gaussians import GaussianField


def test_single_gaussian_peak_and_mass():
    f = GaussianField(2, (64, 64), h=8, nonneg=True)
    p = f.make(torch.tensor([[32.0, 32.0]]), 2.0, torch.tensor([1.0]))
    img = f.render(p)
    assert img.shape == (64, 64)
    assert abs(img[32, 32].item() - 1.0) < 1e-4
    assert abs(img.sum().item() - 2 * math.pi * 2.0 ** 2) < 0.05     # integral of an isotropic 2D Gaussian


def test_render_is_differentiable_3d():
    f = GaussianField(3, (16, 32, 32), h=4, nonneg=True)
    p = f.init_uniform(20, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    f.render(p).sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_periodic_wrap_keeps_mass():
    f = GaussianField(2, (32, 32), h=8, nonneg=False, wrap=True)
    p = f.make(torch.tensor([[0.0, 0.0]]), 2.0, torch.tensor([1.0]))
    assert abs(f.render(p).sum().item() - 2 * math.pi * 4.0) < 0.05
