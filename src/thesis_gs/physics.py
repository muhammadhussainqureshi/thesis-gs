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


def fibonacci_sphere(n, radius, offset=0.0):
    """n nearly uniform points on a sphere (golden-angle spiral); `offset` rotates the spiral so that
    transmitter and receiver sets do not coincide."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n
    r = torch.sqrt(1 - z ** 2)
    phi = math.pi * (3 - math.sqrt(5)) * i + offset
    return radius * torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], -1)   # (n, 3) in (z, y, x)


class Born3D:
    """True 3D linear inverse scattering (first Born approximation) for a D x H x W volume.

    Same voxel size, wavelength (6 voxels) and sensor radius (1.6 m) as the InverseBench 2D setting, but
    with 3D free-space Green's functions G(d) = exp(i k d) / (4 pi d) and point transmitters / receivers
    spread over a full sphere around the volume:
        y[t, r] = k^2 dV * sum_p f_p * u_inc(p, t) * G(|x_r - x_p|),   u_inc(p, t) = G(|x_t - x_p|)
    Precomputes U (P x T) and G (P x R) once (complex64; G is 1.5 GB for 32x128x128 and 360 receivers).
    """

    def __init__(self, shape, receivers, device, transmitters=60, L=0.18, n_xy=128, wave=6, radius=1.6,
                 chunk=65536):
        D, H, W = shape
        dx = L / n_xy
        self.shape, self.k = tuple(shape), 2 * math.pi / (dx * wave)
        self.dV = dx ** 3
        ax = [(torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * dx for n in shape]
        pts = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)              # (P, 3)
        tx = fibonacci_sphere(transmitters, radius)
        rx = fibonacci_sphere(receivers, radius, offset=0.5)

        def green(src):                                   # (P, S) complex64, built in float64 chunks
            out = torch.empty(pts.shape[0], src.shape[0], dtype=torch.complex64, device=device)
            for a in range(0, pts.shape[0], chunk):
                d = torch.cdist(pts[a:a + chunk], src)
                out[a:a + chunk] = (torch.exp(1j * self.k * d) / (4 * math.pi * d)).to(device, torch.complex64)
            return out
        self.U = green(tx)                                 # (P, T) incident fields
        self.G = green(rx)                                 # (P, R) receiver Green's functions
        self.scale = self.k ** 2 * self.dV
        self.n_meas = transmitters * receivers

    def __call__(self, f):                                 # f: (D, H, W) real -> (T, R) complex
        src = f.reshape(-1, 1).to(torch.complex64) * self.U                  # (P, T)
        return self.scale * (src.transpose(0, 1) @ self.G)

    def adjoint_real(self, r):                             # r: (T, R) -> (D, H, W) back-projection
        bp = (self.G.conj() @ r.to(torch.complex64).transpose(0, 1)) * self.U.conj()   # (P, T)
        return (self.scale * bp.sum(-1)).real.view(self.shape)

    def check(self, f, seed=0):
        """Adjoint test: relative mismatch of Re<A f, r> and <f, A* r> (should be ~1e-6)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = torch.complex(torch.randn(self.U.shape[1], self.G.shape[1], generator=g),
                          torch.randn(self.U.shape[1], self.G.shape[1], generator=g)).to(self.U.device)
        lhs = (self(f) * r.conj()).sum().real
        rhs = (f * self.adjoint_real(r)).sum()
        return (abs(lhs - rhs) / abs(lhs).clamp(min=1e-30)).item()
