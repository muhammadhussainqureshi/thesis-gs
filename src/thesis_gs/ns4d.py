"""4D Gaussian Splatting on Navier-Stokes: canonical Gaussians + deformation, training and evaluation.

Frames: every 4th frame from t=2 is held out (test); --val_frames moves further frames out of training for
choosing the smoothness weight (never the test frames). Several smoothness weights (--temporal_reg a b c) and
seeds (--seeds) can be run in one call; --lambda_from picks, per deformation, the weight with the best mean
validation PSNR in an earlier tuning run. Deformation 'interp' is the non-learned baseline: cubic interpolation
of the observed pixels in every training frame, linear interpolation in time for the other frames.

Outputs: results.csv (one row per flow x seed x deformation x weight), per_frame.csv (every frame: PSNR on all
pixels, on the observed pixels and on the unobserved pixels of training frames, enstrophy ratio), spectra.csv
(energy spectra on test frames), summary.csv, reconstructions.png."""
import os
import time

import numpy as np
import torch

from .deform import DeformGRU, DeformMLP, DeformMamba, StaticDeform, apply_deform
from .gaussians import GaussianField
from .metrics import psnr, rel_l2, ssim
from .ns_data import ns_sequences
from .utils import cosine_lr, ensure_inversebench, log

PEAK = 2.0                                      # vorticity / 10 lies in about [-1, 1]


def psnr_region(x, gt, m):
    """PSNR (peak-to-peak 2) over the pixels where m == 1."""
    mse = (((x - gt) ** 2) * m).sum().item() / max(m.sum().item(), 1.0)
    return float(10 * np.log10(PEAK ** 2 / max(mse, 1e-12)))


def energy_spectrum(w):
    """Radially averaged kinetic-energy spectrum of a 2D periodic vorticity field (H, W) on a 2pi box:
    E(k) = sum over the shell |k| ~ k of |w_hat|^2 / (2 |k|^2), k = 1 .. H/2."""
    H, W = w.shape
    wh = torch.fft.fft2(w.double()) / (H * W)
    kx = torch.fft.fftfreq(H, 1.0 / H, dtype=torch.float64, device=w.device)
    ky = torch.fft.fftfreq(W, 1.0 / W, dtype=torch.float64, device=w.device)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    e = (wh.abs() ** 2) / (2 * k2.clamp(min=1.0))
    shell = torch.round(k2.sqrt()).long()
    kmax = H // 2
    E = torch.zeros(kmax + 1, dtype=torch.float64, device=w.device).index_add(
        0, shell.clamp(max=kmax).flatten(), torch.where(shell <= kmax, e, 0.0).flatten())
    return E[1:].cpu().numpy()


def enstrophy(w):
    return 0.5 * float((w.double() ** 2).mean())


def choose_lambda(path, kind):
    """Smoothness weight with the highest mean validation PSNR for this deformation in a tuning results.csv."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df["deform"] == kind]
    if df.empty or "val_frames_psnr" not in df:
        raise ValueError(f"no tuning rows with validation PSNR for '{kind}' in {path}")
    by = df.groupby("temporal_reg")["val_frames_psnr"].mean()
    lam = float(by.idxmax())
    log(f"[ns4d] {kind}: validation PSNR by smoothness weight {by.round(3).to_dict()} -> chose {lam}")
    return lam


def interp_baseline(obs, mask, train_idx, T):
    """Cubic interpolation of the observed pixels per training frame (periodic: points within 16 px of an edge
    are copied across it), then linear interpolation in time for frames that are not training frames."""
    from scipy.interpolate import griddata
    H, W = obs.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    rec = {}
    for k in train_idx:
        m = mask[k].cpu().numpy() > 0
        pts = np.stack(np.nonzero(m), 1).astype(float)
        val = obs[k].cpu().numpy()[m]
        P, V = [pts], [val]
        for dy in (-H, 0, H):
            for dx in (-W, 0, W):
                if dy == 0 and dx == 0:
                    continue
                q = pts + [dy, dx]
                keep = (q[:, 0] > -16) & (q[:, 0] < H + 16) & (q[:, 1] > -16) & (q[:, 1] < W + 16)
                P.append(q[keep])
                V.append(val[keep])
        P, V = np.concatenate(P), np.concatenate(V)
        f = griddata(P, V, (yy, xx), method="cubic")
        if np.isnan(f).any():
            f[np.isnan(f)] = griddata(P, V, (yy[np.isnan(f)], xx[np.isnan(f)]), method="nearest")
        rec[k] = torch.from_numpy(f).float()
    tr = sorted(train_idx)
    out = []
    for k in range(T):
        if k in rec:
            out.append(rec[k])
            continue
        lo = [j for j in tr if j < k]
        hi = [j for j in tr if j > k]
        if lo and hi:
            a, b = lo[-1], hi[0]
            wgt = (k - a) / (b - a)
            out.append((1 - wgt) * rec[a] + wgt * rec[b])
        else:
            out.append(rec[(lo or hi)[-1 if lo else 0]])
    return torch.stack(out).to(obs.device)


def make_net(kind, args):
    return {"mlp": lambda: DeformMLP(width=args.mlp_width),
            "gru": lambda: DeformGRU(hidden=args.gru_hidden),
            "mamba": DeformMamba, "static": StaticDeform,
            "mamba_bi": lambda: DeformMamba(bidirectional=True),
            "mamba_st": lambda: DeformMamba(spatial=True),
            "mamba_vel": lambda: DeformMamba(velocity=True),
            "mamba_st_vel": lambda: DeformMamba(spatial=True, velocity=True)}[kind]()


def train_model(kind, lam, seed, si, obs, mask, train_idx, t_all, field, args, dev):
    """Fit canonical Gaussians + deformation to the observed pixels of the training frames. Returns (rec, net)."""
    T = obs.shape[0]
    torch.manual_seed(si + 1000 * seed)
    net = make_net(kind, args).to(dev)
    seq_net = isinstance(net, (DeformMamba, DeformGRU))
    g0 = torch.Generator(device=dev).manual_seed(si + 1000 * seed)
    canon = field.init_from_field(obs[0] * mask[0], args.n_gauss, scale=2.0, generator=g0)
    canon.requires_grad_(True)
    groups = [{"params": [canon], "lr": args.lr_gauss, "base_lr": args.lr_gauss}]
    if any(True for _ in net.parameters()):
        groups.append({"params": list(net.parameters()), "lr": args.lr_net, "base_lr": args.lr_net})
    opt = torch.optim.Adam(groups)
    for it in range(1, args.iters + 1):
        bidx = train_idx[torch.randint(len(train_idx), (min(args.batch_frames, len(train_idx)),), device=dev)]
        if seq_net:     # one pass over all frames serves the data term and the smoothness term
            full = net(canon[:, :2] / 128.0, t_all, None, full=True)
            delta = full[bidx]
        else:
            delta = net(canon[:, :2] / 128.0, t_all, bidx)
        params = apply_deform(canon, delta)
        pred = torch.stack([field.render(params[b]) for b in range(len(bidx))])
        m = mask[bidx]
        loss = (((pred - obs[bidx]) ** 2) * m).sum() / m.sum().clamp(min=1)
        if lam and kind != "static":
            # smooth motion: offsets at t and t+1 should be close (also covers frames without observations)
            nxt = (bidx + 1).clamp(max=T - 1)
            delta_next = full[nxt] if seq_net else net(canon[:, :2] / 128.0, t_all, nxt)
            loss = loss + lam * ((delta_next - delta) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cosine_lr(opt, None, it, args.iters)
        if it % max(args.iters // 4, 1) == 0 or it == args.iters:
            log(f"    iter {it:5d}  loss {loss.item():.3e}")
    with torch.no_grad():
        allp = apply_deform(canon, net(canon[:, :2] / 128.0, t_all, torch.arange(T, device=dev)))
        rec = torch.stack([field.render(allp[k]) for k in range(T)])
    return rec, net, canon


def run_ns4d(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(args.out, "ns4d")
    os.makedirs(out, exist_ok=True)
    dev = args.device
    ensure_inversebench(args.ib_dir)
    seqs = ns_sequences(args, dev)
    T = seqs[0].shape[0]
    t_all = torch.linspace(0, 1, T, device=dev)
    test = list(range(2, T, 4))                                        # held-out (test) time steps
    val = [k for k in (args.val_frames or []) if k < T]
    assert not set(val) & set(test), "validation frames must not be test frames"
    train = [k for k in range(T) if k not in set(test) | set(val)]
    train_idx = torch.tensor(train, device=dev)
    ftype = {k: "test" if k in test else "val" if k in val else "train" for k in range(T)}
    field = GaussianField(2, (128, 128), h=args.window, nonneg=False, wrap=True, device=dev)  # periodic domain
    lams = {kind: ([choose_lambda(args.lambda_from, kind)] if args.lambda_from and kind not in ("static", "interp")
                   else [0.0] if kind in ("static", "interp") else args.temporal_reg) for kind in args.deform}
    seeds = [0] if args.seeds is None else args.seeds
    log(f"[ns4d] {len(seqs)} flows x {T} frames; train {train}, val {val}, test {test}; "
        f"observed pixel ratios {args.obs_ratios}; deformations {args.deform}; weights {lams}; seeds {seeds}")

    rows, frame_rows, spec_rows, examples = [], [], [], {}
    for ratio in args.obs_ratios:
        for si, seq in enumerate(seqs):
            gen = torch.Generator(device=dev).manual_seed(500 + si)
            mask = (torch.rand(seq.shape, generator=gen, device=dev) < ratio).float()   # observed pixels per frame
            obs = seq + args.ns_noise * torch.randn(seq.shape, generator=gen, device=dev)
            spec_gt = np.mean([energy_spectrum(seq[k]) for k in test], 0)
            for kind in args.deform:
                for lam in lams[kind]:
                    for seed in (seeds if kind != "interp" else [0]):
                        log(f"[ns4d] ratio {ratio}, flow {si}, deform={kind}, lambda={lam}, seed={seed}")
                        t0 = time.time()
                        if kind == "interp":
                            rec, net, canon = interp_baseline(obs, mask, train, T), None, None
                        else:
                            rec, net, canon = train_model(kind, lam, seed, si, obs, mask, train_idx, t_all,
                                                          field, args, dev)
                        train_time = time.time() - t0
                        key = dict(obs_ratio=ratio, sample=si, deform=kind, temporal_reg=lam, seed=seed)
                        res = {}
                        for name, idx in [("train_frames", train), ("val_frames", val), ("heldout_frames", test)]:
                            if not idx:
                                continue
                            res[f"{name}_rel_l2"] = float(np.mean([rel_l2(rec[k], seq[k]) for k in idx]))
                            res[f"{name}_psnr"] = float(np.mean([psnr(rec[k], seq[k], PEAK) for k in idx]))
                            res[f"{name}_ssim"] = float(np.mean([ssim(rec[k], seq[k], PEAK) for k in idx]))
                        # error split: observed pixels (training fit) / unobserved pixels of training frames
                        res["train_obs_psnr"] = float(np.mean([psnr_region(rec[k], seq[k], mask[k]) for k in train]))
                        res["train_unobs_psnr"] = float(np.mean([psnr_region(rec[k], seq[k], 1 - mask[k])
                                                                 for k in train]))
                        for k in range(T):
                            frame_rows.append({**key, "frame": k, "frame_type": ftype[k],
                                               "psnr_all": psnr(rec[k], seq[k], PEAK),
                                               "psnr_obs": psnr_region(rec[k], seq[k], mask[k])
                                               if ftype[k] == "train" else np.nan,
                                               "psnr_unobs": psnr_region(rec[k], seq[k], 1 - mask[k])
                                               if ftype[k] == "train" else np.nan,
                                               "enstrophy_ratio": enstrophy(rec[k]) / enstrophy(seq[k])})
                        spec = np.mean([energy_spectrum(rec[k]) for k in test], 0)
                        kk = np.arange(1, len(spec) + 1)
                        good = (kk <= 32) & (spec_gt > 1e-8 * spec_gt.max())          # resolved, non-negligible shells
                        res["spec_log_err"] = float(np.mean(np.abs(np.log10(np.maximum(spec[good], 1e-30)
                                                                            / spec_gt[good]))))
                        res["enstrophy_ratio_test"] = float(np.mean([enstrophy(rec[k]) / enstrophy(seq[k])
                                                                     for k in test]))
                        spec_rows += [{**key, "k": int(k), "E_rec": float(e), "E_gt": float(g)}
                                      for k, e, g in zip(kk, spec, spec_gt)]
                        n_net = sum(p.numel() for p in net.parameters()) if net is not None else 0
                        rows.append({**key, **res, "n_gauss": canon.shape[0] if canon is not None else 0,
                                     "net_params": n_net, "time_s": round(train_time, 1)})
                        log(f"    -> PSNR test {res['heldout_frames_psnr']:.2f} dB"
                            + (f" | val {res['val_frames_psnr']:.2f}" if val else "")
                            + f" | observed px {res['train_obs_psnr']:.2f} | unobserved px "
                              f"{res['train_unobs_psnr']:.2f} | {n_net} net params ({train_time:.0f}s)")
                        if si == 0 and seed == seeds[0]:
                            examples[(ratio, kind, lam)] = rec.cpu()
                            if net is not None:
                                torch.save({"canonical": canon.detach().cpu(), "net": net.state_dict()},
                                           os.path.join(out, f"model_ratio{ratio}_{kind}_lam{lam}_flow0.pt"))
                        # write after every model, so a time-out keeps what is done
                        pd.DataFrame(rows).to_csv(os.path.join(out, "results.csv"), index=False)
                        pd.DataFrame(frame_rows).to_csv(os.path.join(out, "per_frame.csv"), index=False)
                        pd.DataFrame(spec_rows).to_csv(os.path.join(out, "spectra.csv"), index=False)

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.endswith(("rel_l2", "psnr", "ssim", "err", "test"))] + ["net_params", "time_s"]
    grp = df.groupby(["obs_ratio", "deform", "temporal_reg"])
    summ = grp[cols].mean().join(grp[["heldout_frames_psnr"]].std().add_suffix("_std"))
    summ["n_runs"] = grp.size()
    summ = summ.round(4)
    summ.to_csv(os.path.join(out, "summary.csv"))
    log(f"[ns4d] summary\n{summ.to_string()}")

    # figure: flow 0, a few frames incl. test ones, for every (ratio, deform, weight)
    show = sorted(set([0, test[0], T // 2, test[-1], T - 1]))
    keys = list(examples.keys())
    fig, ax = plt.subplots(1 + len(keys), len(show), figsize=(2.2 * len(show), 2.2 * (1 + len(keys))), squeeze=False)
    for j, k in enumerate(show):
        tag = " (test)" if k in test else ""
        ax[0, j].imshow(seqs[0][k].cpu(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0, j].set_title(f"GT t={k}{tag}", fontsize=7)
        for i, key in enumerate(keys):
            ax[i + 1, j].imshow(examples[key][k], cmap="RdBu_r", vmin=-1, vmax=1)
            ax[i + 1, j].set_title(f"{key[1]} λ={key[2]}, obs {key[0]:.0%}", fontsize=7)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reconstructions.png"), dpi=130)
    plt.close(fig)
    return summ
