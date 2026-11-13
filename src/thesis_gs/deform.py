"""Deformation networks over time for 4D Gaussians: MLP (4DGS-style), Mamba (selective SSM), static."""
import math

import torch
import torch.nn as nn

from .ssm import MambaBlock, PointMambaEncoder, farthest_point_sampling, serial_orders



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


class DeformMamba(nn.Module):
    """Deformation as a sequence model over time: each Gaussian's trajectory is one sequence of length T,
    processed by Mamba blocks, so the offset at frame t depends on the Gaussian's state at earlier frames.

    spatial=True adds a PointMamba-style spatial scan (space-time Mamba): the Gaussians are grouped into patches
    around farthest-point-sampled key points, the patch tokens of every frame are serialised along Hilbert and
    Trans-Hilbert curves and processed by Mamba, and each Gaussian receives its patch's context, so Gaussians that
    move together (a vortex) share information. velocity=True predicts a per-frame velocity and integrates it in
    time (position offset at t = sum of velocities before t), instead of an independent offset per frame."""

    def __init__(self, n_out=6, n_freq=6, d_model=64, n_layers=2, bidirectional=False, spatial=False,
                 velocity=False, n_patches=64, grid=(128, 128)):
        super().__init__()
        self.n_freq, self.velocity, self.n_patches, self.grid = n_freq, velocity, n_patches, grid
        self.inp = nn.Linear(3 * 2 * n_freq, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        # bidirectional: a second stack reads the trajectory backwards in time (future -> past)
        self.blocks_bwd = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)]) if bidirectional else None
        if spatial:
            self.sp_in = nn.Linear(d_model, d_model)
            self.sp_pos = nn.Linear(2 * 2 * n_freq, d_model)
            # lighter blocks: the spatial scan runs for every frame at once (T sequences of 2 x n_patches tokens)
            self.sp_enc = PointMambaEncoder(d_model, n_layers=n_layers, n_orders=2, d_state=8, expand=1)
            self.sp_out = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.sp_out.weight)
            nn.init.zeros_(self.sp_out.bias)
        self.spatial = spatial
        self._patches = None
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _make_patches(self, mu_norm):
        pts = mu_norm.detach() * torch.tensor(self.grid, device=mu_norm.device, dtype=mu_norm.dtype)
        key = farthest_point_sampling(pts, self.n_patches)
        assign = torch.cdist(pts, pts[key]).argmin(1)
        orders = serial_orders(pts[key], self.grid, "pointmamba")
        self._patches = (key, assign, orders, pts.shape[0])

    def forward(self, mu_norm, t_all, idx, full=False):
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
        if self.spatial:
            if self._patches is None or self._patches[3] != N:
                self._make_patches(mu_norm)
            key, assign, orders, _ = self._patches
            M = key.shape[0]
            f = self.sp_in(h).transpose(0, 1)                                 # (T, N, d)
            cnt = torch.zeros(M, device=h.device).index_add(0, assign, torch.ones(N, device=h.device))
            tok = torch.zeros(T, M, f.shape[-1], device=h.device).index_add(1, assign, f) / cnt[None, :, None]
            tok = tok + self.sp_pos(fourier(mu_norm[key].detach(), self.n_freq))[None]
            ctx = self.sp_enc(tok, orders)                                    # (T, M, d)
            h = h + self.sp_out(ctx[:, assign].transpose(0, 1))              # each Gaussian gets its patch context
        out = self.head(self.norm(h))                                         # (N, T, n_out)
        if self.velocity:                                                     # integrate velocities over time
            pos = torch.cumsum(out[:, :, :2], 1) * 0.25
            pos = torch.cat([torch.zeros_like(pos[:, :1]), pos[:, :-1]], 1)
            out = torch.cat([pos, out[:, :, 2:]], -1)
        out = out.transpose(0, 1)                                             # (T, N, n_out)
        return out if full else out[idx]


class StaticDeform(nn.Module):
    """No deformation (ablation): the same Gaussians for every frame."""

    def forward(self, mu_norm, t_all, idx):
        return torch.zeros(len(idx), mu_norm.shape[0], 6, device=mu_norm.device)


def apply_deform(canon, delta, move_scale=8.0):
    """canon (N,6) + delta (B,N,6) -> per-frame params (B,N,6). Positions move in pixels."""
    d = delta.clone()
    d[..., :2] = d[..., :2] * move_scale
    return canon[None] + d
