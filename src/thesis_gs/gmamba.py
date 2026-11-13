"""PointMamba over Gaussians: a Mamba network that treats the Gaussians of one scene as a point cloud and
predicts a correction for every Gaussian from its spatial neighbourhood (Liang et al., PointMamba, NeurIPS 2024).

Per-scene use (no training data): the network is fitted together with the Gaussians by the physics loss, like
the 4D deformation network; it starts as the identity (zero-initialised head) and couples neighbouring
Gaussians, acting as a learned spatial prior.

Pipeline: per-Gaussian embedding (Fourier features of the centre + the other parameters [+ the back-projected
residual at the centre]) -> patches around farthest-point-sampled key points (mean-pooled, a mini-PointNet) ->
serialisation along Hilbert / Trans-Hilbert curves with an order indicator -> Mamba blocks -> each Gaussian gets
its patch context -> head -> parameter offsets."""
import torch
import torch.nn as nn

from .deform import fourier
from .ssm import PointMambaEncoder, farthest_point_sampling, serial_orders

MODES = ["mlp", "random", "hilbert", "pointmamba"]


class GaussPointMamba(nn.Module):
    def __init__(self, dim, n_param, grid, mode="pointmamba", d_model=64, n_layers=2, n_patches=None,
                 n_freq=4, residual_feat=False, pos_scale=1.0):
        super().__init__()
        assert mode in MODES, mode
        self.dim, self.P, self.grid, self.mode = dim, n_param, tuple(grid), mode
        self.n_freq, self.residual_feat, self.pos_scale = n_freq, residual_feat, pos_scale
        self.n_patches = n_patches or (256 if dim == 3 else 128)
        d_in = dim * 2 * n_freq + (n_param - dim) + (1 if residual_feat else 0)
        self.embed = nn.Sequential(nn.Linear(d_in, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if mode != "mlp":
            self.tok = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
            self.key_pos = nn.Linear(dim * 2 * n_freq, d_model)
            self.enc = PointMambaEncoder(d_model, n_layers=n_layers, n_orders=2 if mode == "pointmamba" else 1)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_param))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("gsize", torch.tensor(self.grid, dtype=torch.float32))
        self._patches = None
        self._gen = torch.Generator().manual_seed(0)

    def update_patches(self, p):
        """(Re)build key points, patch assignment and serialisation; call after densification / pruning."""
        if self.mode == "mlp":
            return
        mu = p[:, :self.dim].detach()
        key = farthest_point_sampling(mu, self.n_patches)
        assign = torch.cdist(mu, mu[key]).argmin(1)
        orders = serial_orders(mu[key].clamp(min=0), self.grid,
                               {"random": "random", "hilbert": "hilbert", "pointmamba": "pointmamba"}[self.mode],
                               generator=self._gen)
        self._patches = (key, assign, orders, p.shape[0])

    def n_params(self):
        return sum(q.numel() for q in self.parameters())

    def forward(self, p, resid=None):
        """p: (N, P) Gaussian parameters -> p + delta(p). resid: residual field on the grid (or None)."""
        mu = p[:, :self.dim].detach()
        mu_n = (mu / self.gsize).clamp(0, 1)
        feats = [fourier(mu_n, self.n_freq), p[:, self.dim:].detach()]
        if self.residual_feat:
            if resid is None:
                feats.append(torch.zeros_like(mu[:, :1]))
            else:
                ii = mu.round().long()
                for k in range(self.dim):
                    ii[:, k] = ii[:, k].clamp(0, self.grid[k] - 1)
                feats.append(resid[tuple(ii.unbind(1))][:, None])
        e = self.embed(torch.cat(feats, -1))                                # (N, d)
        if self.mode == "mlp":
            ctx = torch.zeros_like(e)
        else:
            if self._patches is None or self._patches[3] != p.shape[0]:
                self.update_patches(p)
            key, assign, orders, _ = self._patches
            M = key.shape[0]
            f = self.tok(e)
            cnt = torch.zeros(M, device=e.device).index_add(0, assign, torch.ones(len(assign), device=e.device))
            tok = torch.zeros(M, f.shape[-1], device=e.device).index_add(0, assign, f) / cnt.clamp(min=1)[:, None]
            tok = tok + self.key_pos(fourier(mu_n[key], self.n_freq))
            ctx = self.enc(tok[None], orders)[0][assign]                    # each Gaussian gets its patch context
        delta = self.head(torch.cat([e, ctx], -1))
        delta = torch.cat([delta[:, :self.dim] * self.pos_scale, delta[:, self.dim:]], -1)
        return p + delta
