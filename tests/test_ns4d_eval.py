import math

import torch

from thesis_gs.deform import DeformGRU, DeformMamba, DeformMLP
from thesis_gs.ns4d import energy_spectrum, interp_baseline


def test_spectrum_single_mode_parseval():
    x = torch.linspace(0, 2 * math.pi, 129)[:-1]
    X, _ = torch.meshgrid(x, x, indexing="ij")
    E = energy_spectrum(torch.cos(3 * X))
    assert E.argmax() + 1 == 3
    assert abs(E.sum() - 0.25 / 9) < 1e-6           # 0.5 <u^2> for w = cos(3x)


def test_interp_exact_on_observed_and_linear_in_time():
    torch.manual_seed(0)
    T = 5
    obs = torch.randn(T, 32, 32)
    mask = (torch.rand(T, 32, 32) < 0.3).float()
    rec = interp_baseline(obs, mask, [0, 1, 3, 4], T)
    assert torch.allclose(rec[1][mask[1] > 0], obs[1][mask[1] > 0], atol=1e-4)
    assert torch.allclose(rec[2], 0.5 * (rec[1] + rec[3]))


def test_matched_network_sizes():
    n = lambda m: sum(p.numel() for p in m.parameters())
    ref = n(DeformMamba())
    for m in (DeformMLP(width=175), DeformGRU(hidden=103)):
        assert abs(n(m) / ref - 1) < 0.10
