"""Selective state-space (Mamba) block with a parallel scan, plus point-cloud serialisation helpers
(farthest-point sampling, Hilbert curves) used by the PointMamba-style modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------- scan
def scan_sequential(a, b):
    """Reference: h_t = a_t * h_{t-1} + b_t along dim 1, h_{-1} = 0."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)


def scan_parallel(a, b):
    """Same recurrence as scan_sequential, computed with the Hillis-Steele doubling scan in ceil(log2 L)
    vectorised steps. Only products of factors in (0, 1] occur, so it is numerically stable."""
    L = b.shape[1]
    off = 1
    while off < L:
        a_prev = F.pad(a[:, :-off], (0,) * (2 * (a.dim() - 2)) + (off, 0), value=1.0)
        b_prev = F.pad(b[:, :-off], (0,) * (2 * (b.dim() - 2)) + (off, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        off *= 2
    return b


class MambaBlock(nn.Module):
    """Minimal pure-PyTorch Mamba block (Gu & Dao 2023), causal over the sequence, with a parallel scan."""

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, parallel="auto"):
        super().__init__()
        di = expand * d_model
        self.di, self.ds, self.parallel = di, d_state, parallel
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
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.ds, self.ds], -1)
        dt = F.softplus(self.dt_proj(dt))                     # (B, L, di)
        A = -torch.exp(self.A_log)                            # (di, ds)
        a = torch.exp(dt[..., None] * A)                      # (B, L, di, ds)
        b = dt[..., None] * Bm[:, :, None, :] * u[..., None]  # (B, L, di, ds)
        # parallel scan for long sequences (spatial scans); a short loop is cheaper in memory for short ones (time)
        use_par = self.parallel if isinstance(self.parallel, bool) else L > 32
        h = scan_parallel(a, b) if use_par else scan_sequential(a, b)
        y = (h * Cm[:, :, None, :]).sum(-1)
        y = (y + self.D * u) * F.silu(z)
        return x + self.out_proj(y)


# -------------------------------------------------------------------------------------- serialisation
def hilbert_index(coords, bits):
    """Hilbert-curve index of integer coordinates (N, D) in [0, 2^bits), D = 2 or 3 (Skilling's algorithm).
    Consecutive indices are neighbouring cells, so sorting by it keeps spatial neighbours together."""
    x = coords.long().clone()
    n_dim = x.shape[1]
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:                                              # inverse undo excess work
        P = Q - 1
        for i in range(n_dim):
            hi = (x[:, i] & Q) != 0
            x0 = x[:, 0].clone()
            x[:, 0] = torch.where(hi, x0 ^ P, x0)
            t = (x[:, 0] ^ x[:, i]) & P
            t = torch.where(hi, torch.zeros_like(t), t)
            x[:, 0] = x[:, 0] ^ t
            x[:, i] = x[:, i] ^ t
        Q >>= 1
    for i in range(1, n_dim):                                 # Gray encode
        x[:, i] ^= x[:, i - 1]
    t = torch.zeros_like(x[:, 0])
    Q = M
    while Q > 1:
        t = torch.where((x[:, n_dim - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    x = x ^ t[:, None]
    idx = torch.zeros_like(x[:, 0])                           # interleave the transposed bits
    for b in range(bits - 1, -1, -1):
        for i in range(n_dim):
            idx = (idx << 1) | ((x[:, i] >> b) & 1)
    return idx


def farthest_point_sampling(pts, m, generator=None):
    """Indices of m points chosen by farthest-point sampling (pts: (N, D))."""
    N = pts.shape[0]
    m = min(m, N)
    idx = torch.empty(m, dtype=torch.long, device=pts.device)
    first = torch.randint(N, (1,), generator=generator, device=pts.device) if generator is not None else \
        torch.zeros(1, dtype=torch.long, device=pts.device)
    idx[0] = first
    d = ((pts - pts[first]) ** 2).sum(-1)
    for k in range(1, m):
        idx[k] = torch.argmax(d)
        d = torch.minimum(d, ((pts - pts[idx[k]]) ** 2).sum(-1))
    return idx


def serial_orders(key_pts, grid, mode, generator=None):
    """Orderings of key points as index tensors. mode: 'hilbert' -> [Hilbert],
    'pointmamba' -> [Hilbert, Trans-Hilbert (axes reversed)], 'random' -> [random permutation]."""
    n = key_pts.shape[0]
    if mode == "random":
        return [torch.randperm(n, generator=generator, device="cpu").to(key_pts.device)]
    bits = max(1, math.ceil(math.log2(max(grid))))
    scale = torch.tensor([(1 << bits) / g for g in grid], device=key_pts.device)
    q = (key_pts * scale).floor().clamp(0, (1 << bits) - 1).long()
    orders = [torch.argsort(hilbert_index(q, bits))]
    if mode == "pointmamba":
        orders.append(torch.argsort(hilbert_index(q.flip(1), bits)))
    return orders


class PointMambaEncoder(nn.Module):
    """PointMamba-style encoder over a set of tokens with positions: serialise the tokens along one or two
    space-filling curves, add a learned order indicator (scale, shift) per ordering, run Mamba blocks over the
    concatenated sequences, and return per-token features averaged over the orderings (original order)."""

    def __init__(self, d_model, n_layers=2, n_orders=2, d_state=16, expand=2):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state, expand=expand) for _ in range(n_layers)])
        self.gamma = nn.Parameter(torch.ones(n_orders, d_model))
        self.beta = nn.Parameter(torch.zeros(n_orders, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tok, orders):                           # tok (B, M, d); orders: list of (M,) index tensors
        seqs = [tok[:, o] * self.gamma[k] + self.beta[k] for k, o in enumerate(orders)]
        h = torch.cat(seqs, 1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        M = tok.shape[1]
        out = torch.zeros_like(tok)
        for k, o in enumerate(orders):
            out = out.index_add(1, o, h[:, k * M:(k + 1) * M])
        return out / len(orders)
