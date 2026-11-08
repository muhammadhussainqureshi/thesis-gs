import math

import torch

from thesis_gs.gaussians import GaussianField
from thesis_gs.physics import Born3D, fibonacci_sphere


def test_born3d_point_scatterer_and_adjoint():
    shape = (4, 16, 16)
    op = Born3D(shape, 12, "cpu", transmitters=6)
    f = torch.zeros(shape)
    f[1, 5, 9] = 1.0
    dx = 0.18 / 128
    k = 2 * math.pi / (6 * dx)
    p = torch.tensor([(1 - 1.5) * dx, (5 - 7.5) * dx, (9 - 7.5) * dx], dtype=torch.float64)
    dt = (fibonacci_sphere(6, 1.6) - p).norm(dim=1)
    dr = (fibonacci_sphere(12, 1.6, 0.5) - p).norm(dim=1)
    g = lambda d: torch.exp(1j * k * d) / (4 * math.pi * d)
    expected = k ** 2 * dx ** 3 * g(dt)[:, None] * g(dr)[None]
    assert ((op(f).to(torch.complex128) - expected).abs().max() / expected.abs().max()) < 1e-5
    assert op.check(torch.rand(shape)) < 1e-4


def test_generalised_gaussian_is_flatter():
    f1 = GaussianField(2, (32, 32), h=8, beta=1.0)
    f4 = GaussianField(2, (32, 32), h=8, beta=4.0)
    p = f1.make(torch.tensor([[16.0, 16.0]]), 2.0, torch.tensor([1.0]))
    a, b = f1.render(p), f4.render(p)
    assert torch.isclose(a[16, 16], b[16, 16])            # same peak
    assert b[16, 17] > a[16, 17] and b[16, 21] < a[16, 21]   # flatter top, sharper fall-off
