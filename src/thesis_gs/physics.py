"""Linear inverse-scattering forward operator (InverseBench, Born approximation), applied per slice."""
import math

import torch


class SliceScatter:
    """InverseBench linear inverse-scattering operator, applied to every 128x128 slice of a stack.
        y[z, t, r] = dx*dy * sum_p f[z, p] * conj(uinc[p, t]) * G[p, r]
    Identical to InverseScatter.forward(f, unnormalize=False) per slice (checked at start-up)."""

    def __init__(self, receivers, device):
        from inverse_problems.inverse_scatter import InverseScatter
        self.ib = InverseScatter(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=receivers, numTrans=20,
                                 sensorRadius=1.6, sigma_noise=1e-4, unnorm_shift=1.0, unnorm_scale=0.5,
                                 svd=False, device=device)
        self.dxdy = self.ib.dx * self.ib.dy
        self.uc = torch.conj(self.ib.uinc_dom_set[..., 0].reshape(-1, 20)).to(device, torch.complex64)   # (P, T)
        self.G = self.ib.sensor_greens_function_set[..., 0].reshape(-1, receivers).to(device, torch.complex64)  # (P, R)

    def __call__(self, f):                       # f: (..., 128, 128) real -> (Z, T, R) complex
        f = f.reshape(-1, 128 * 128).to(torch.complex64)
        src = f[:, :, None] * self.uc[None]                      # (Z, P, T)
        return self.dxdy * torch.einsum("zpt,pr->ztr", src, self.G)

    def adjoint_real(self, r):                   # r: (Z, T, R) -> (Z, 128, 128) back-projection
        bp = torch.einsum("pr,ztr->zpt", self.G.conj(), r.to(torch.complex64)) * self.uc.conj()[None]
        return (self.dxdy * bp.sum(-1)).real.view(-1, 128, 128)

    def check(self, f2d):
        y_ib = self.ib.forward(f2d[None, None].double(), unnormalize=False)[0]
        return ((self(f2d)[0] - y_ib.to(torch.complex64)).abs().max() / y_ib.abs().max()).item()


def add_noise(y, sigma, gen):
    n = torch.randn(y.shape, generator=gen, device=y.device, dtype=torch.float32) \
        + 1j * torch.randn(y.shape, generator=gen, device=y.device, dtype=torch.float32)
    return y + sigma * n / math.sqrt(2)
