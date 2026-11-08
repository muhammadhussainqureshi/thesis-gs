"""Anisotropic 2D/3D Gaussian field rendered on a regular grid."""
import math

import numpy as np
import torch
import torch.nn.functional as F



def quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


class GaussianField:
    """N anisotropic Gaussians on a regular grid, one parameter tensor of shape (N, P).

    2D layout: [mu(2), log_s(2), theta(1), amp(1)]            P = 6
    3D layout: [mu(3), log_s(3), quat(4), amp(1)]             P = 11
    Coordinates are in grid-index units, in array axis order ((y,x) or (z,y,x)).
    amp is softplus(raw) >= 0 when nonneg=True (permittivity), raw signed otherwise (vorticity).
    Each Gaussian is evaluated only in a local window of +-h cells around its centre (the "3 sigma"
    trick from 3DGS), so cost is O(N * (2h+1)^dim), independent of the grid size.
    beta = 1 gives ordinary Gaussians exp(-q/2); beta > 1 gives generalised ("super-") Gaussians
    exp(-q^beta/2) with a flatter top and sharper edges, for piecewise-constant objects.
    """

    def __init__(self, dim, grid, h, nonneg=True, wrap=False, device="cpu", beta=1.0):
        if beta < 1:
            raise ValueError("beta must be >= 1")
        self.dim, self.grid, self.h, self.nonneg, self.wrap = dim, tuple(grid), h, nonneg, wrap
        self.beta = float(beta)
        self.P = 6 if dim == 2 else 11
        self.device = device
        r = torch.arange(-h, h + 1, device=device)
        self.offsets = torch.stack(torch.meshgrid(*([r] * dim), indexing="ij"), -1).reshape(-1, dim)  # (K, dim)
        self.s_min, self.s_max = 0.35, h / 3.0

    # ---- parameter helpers
    def make(self, mu, scale, amp):
        n = mu.shape[0]
        log_s = torch.full((n, self.dim), math.log(scale), device=mu.device)
        if self.dim == 2:
            rot = torch.zeros(n, 1, device=mu.device)
        else:
            rot = torch.zeros(n, 4, device=mu.device)
            rot[:, 0] = 1.0
        a = torch.log(torch.expm1(amp.clamp(min=1e-4))) if self.nonneg else amp
        return torch.cat([mu, log_s, rot, a[:, None]], 1)

    def amplitude(self, p):
        return F.softplus(p[:, -1]) if self.nonneg else p[:, -1]

    def init_from_field(self, field, n, scale=2.0, generator=None):
        """Sample centres proportional to |field| (+10% uniform)."""
        w = field.abs().flatten()
        w = w + 0.1 * w.mean().clamp(min=1e-6)
        idx = torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)
        coords = torch.stack(torch.unravel_index(idx, self.grid), -1).float()
        coords = coords + torch.rand(coords.shape, generator=generator, device=coords.device) - 0.5
        amp = field.flatten()[idx] * 0.5
        if self.nonneg:
            amp = amp.abs().clamp(min=0.02)
        return self.make(coords, scale, amp)

    def init_uniform(self, n, scale=3.0, amp=0.05, generator=None):
        g = torch.tensor(self.grid, device=self.device, dtype=torch.float32)
        mu = torch.rand(n, self.dim, generator=generator, device=self.device) * (g - 1)
        return self.make(mu, scale, torch.full((n,), amp, device=self.device))

    # ---- rendering
    def inv_cov(self, p):
        d = self.dim
        s = torch.exp(p[:, d:2 * d]).clamp(self.s_min, self.s_max)
        if d == 2:
            th = p[:, 4]
            c, sn = torch.cos(th), torch.sin(th)
            R = torch.stack([c, -sn, sn, c], -1).view(-1, 2, 2)
        else:
            R = quat_to_rot(p[:, 6:10])
        return R @ torch.diag_embed(1.0 / s ** 2) @ R.transpose(1, 2)

    def render(self, p):
        d = self.dim
        mu = p[:, :d]
        base = torch.round(mu).detach().long()
        coords = base[:, None, :] + self.offsets[None]                       # (N, K, d)
        diff = coords.float() - mu[:, None, :]
        q = torch.einsum("nkd,nde,nke->nk", diff, self.inv_cov(p), diff)
        if self.beta != 1.0:
            q = q.clamp(min=0) ** self.beta
        val = self.amplitude(p)[:, None] * torch.exp(-0.5 * q)
        g = torch.tensor(self.grid, device=p.device)
        if self.wrap:
            coords = coords % g
        else:
            inside = ((coords >= 0) & (coords < g)).all(-1)
            val = val * inside
            coords = torch.minimum(coords.clamp(min=0), g - 1)
        strides = torch.tensor([int(np.prod(self.grid[i + 1:])) for i in range(d)], device=p.device)
        flat = (coords * strides).sum(-1)
        out = torch.zeros(int(np.prod(self.grid)), device=p.device, dtype=val.dtype)
        out = out.index_add(0, flat.flatten(), val.flatten())
        return out.view(self.grid)
