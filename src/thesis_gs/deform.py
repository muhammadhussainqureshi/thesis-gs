"""Deformation networks over time for 4D Gaussians: MLP (4DGS-style), Mamba (selective SSM), static."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def fourier(x, n_freq):
    """x in [0,1] -> [sin(2^k pi x), cos(2^k pi x)]_k  (periodic-friendly)."""
    f = (2.0 ** torch.arange(n_freq, device=x.device)) * math.pi
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class DeformMLP(nn.Module):
    """4DGS-style: (canonical Gaussian position, time) -> parameter offsets, independently per (Gaussian, t)."""

    def __init__(self, n_out=6, n_freq=6, width=128, depth=3):
        super().__init__()
        self.n_freq = n_freq
        layers, d_in = [], 3 * 2 * n_freq
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]
            d_in = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mu_norm, t_all, idx):
        """mu_norm (N,2) in [0,1], t_all (T,) in [0,1], idx: frame indices -> (len(idx), N, n_out)."""
        t = t_all[idx]
        N, B = mu_norm.shape[0], t.shape[0]
        x = torch.cat([fourier(mu_norm, self.n_freq)[None].expand(B, N, -1),
                       fourier(t[:, None, None].expand(B, N, 1), self.n_freq)], -1)
        return self.head(self.body(x))


class MambaBlock(nn.Module):
    """Minimal pure-PyTorch Mamba (selective state-space) block, Gu & Dao 2023. Causal over the sequence."""

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        di = expand * d_model
        self.di, self.ds = di, d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model)

    def forward(self, x):                                     # x: (B, L, d)
        B, L, _ = x.shape
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.ds, self.ds], -1)
        dt = F.softplus(self.dt_proj(dt))                     # (B, L, di)
        A = -torch.exp(self.A_log)                            # (di, ds)
        h = torch.zeros(B, self.di, self.ds, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):                                    # selective scan
            h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = (torch.stack(ys, 1) + self.D * u) * F.silu(z)
        return x + self.out_proj(y)


class DeformMamba(nn.Module):
    """Deformation as a sequence model over time: each Gaussian's trajectory is one sequence of length T,
    processed by Mamba blocks, so the offset at frame t depends on the Gaussian's state at earlier frames."""

    def __init__(self, n_out=6, n_freq=6, d_model=64, n_layers=2, bidirectional=False):
        super().__init__()
        self.n_freq = n_freq
        self.inp = nn.Linear(3 * 2 * n_freq, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        # bidirectional: a second stack reads the trajectory backwards in time (future -> past)
        self.blocks_bwd = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)]) if bidirectional else None
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mu_norm, t_all, idx):
        N, T = mu_norm.shape[0], t_all.shape[0]
        x = torch.cat([fourier(mu_norm, self.n_freq)[:, None].expand(N, T, -1),
                       fourier(t_all[None, :, None].expand(N, T, 1), self.n_freq)], -1)
        h0 = self.inp(x)
        h = h0
        for b in self.blocks:
            h = b(h)
        if self.blocks_bwd is not None:
            hb = torch.flip(h0, [1])
            for b in self.blocks_bwd:
                hb = b(hb)
            h = h + torch.flip(hb, [1])
        out = self.head(self.norm(h))                         # (N, T, n_out)
        return out[:, idx].transpose(0, 1)                    # (len(idx), N, n_out)


class StaticDeform(nn.Module):
    """No deformation (ablation): the same Gaussians for every frame."""

    def forward(self, mu_norm, t_all, idx):
        return torch.zeros(len(idx), mu_norm.shape[0], 6, device=mu_norm.device)


def apply_deform(canon, delta, move_scale=8.0):
    """canon (N,6) + delta (B,N,6) -> per-frame params (B,N,6). Positions move in pixels."""
    d = delta.clone()
    d[..., :2] = d[..., :2] * move_scale
    return canon[None] + d
