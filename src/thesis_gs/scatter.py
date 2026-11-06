"""Inverse scattering with Gaussians (2D and 3D): direct fit, physics fit, pixel/voxel baseline, experiment runner."""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianField
from .metrics import psnr, rel_l2, ssim
from .physics import SliceScatter, add_noise
from .utils import cosine_lr, ensure_inversebench, ensure_lmdb_dataset, log, read_lmdb_images
from .volumes import load_volumes, synthetic_nuclei_volume


def fit_scatter(gt, field, mode, op=None, y=None, n_init=400, n_max=4000, iters=3000, lr=0.05,
                densify_every=250, n_add=None, densify_until=0.6, l1=0.0, seed=0, log_every=500):
    """mode='direct': MSE to ground truth.  mode='physics': ||A f - y||^2/||y||^2 (gt only for metrics)."""
    g = torch.Generator(device=gt.device).manual_seed(seed)
    n_add = n_add or max(50, n_init // 4)
    p = (field.init_from_field(gt, n_init, generator=g) if mode == "direct"
         else field.init_uniform(n_init, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    ynorm = (y.abs() ** 2).sum() if y is not None else None
    t0 = time.time()
    for it in range(1, iters + 1):
        img = field.render(p)
        loss = F.mse_loss(img, gt) if mode == "direct" else ((op(img) - y).abs() ** 2).sum() / ynorm
        if l1 and mode != "direct":
            loss = loss + l1 * img.mean()          # sparsity: no mass where the measurements don't ask for it
        opt.zero_grad()
        loss.backward()
        opt.step()
        cosine_lr(opt, lr, it, iters)
        if densify_every and it % densify_every == 0 and it <= densify_until * iters:
            with torch.no_grad():
                img = field.render(p)
                if mode == "direct":
                    res = gt - img
                else:
                    res = op.adjoint_real(y - op(img)).view(gt.shape)
                    res = res / res.abs().max().clamp(min=1e-12) * 0.3
                keep = field.amplitude(p) > 1e-3
                k = max(0, min(n_add, n_max - int(keep.sum())))
                new = []
                if k:
                    idx = torch.topk(res.abs().flatten(), k).indices
                    mu = torch.stack(torch.unravel_index(idx, field.grid), -1).float()
                    new = [field.make(mu, 1.5, res.flatten()[idx].abs().clamp(min=0.02))]
                p = torch.cat([p.detach()[keep]] + new, 0).requires_grad_(True)
                cur = opt.param_groups[0]["lr"]
                opt = torch.optim.Adam([p], lr=cur)
                opt.param_groups[0]["base_lr"] = lr
        if it % log_every == 0 or it == iters:
            with torch.no_grad():
                log(f"    iter {it:5d}  loss {loss.item():.3e}  PSNR {psnr(field.render(p).clamp(0, 1), gt):6.2f} dB  #G {p.shape[0]}")
    with torch.no_grad():
        rec = field.render(p).clamp(0, 1)
    return p.detach(), rec, {"psnr": psnr(rec, gt), "ssim": ssim(rec, gt), "rel_l2": rel_l2(rec, gt),
                             "n_gauss": p.shape[0], "time_s": round(time.time() - t0, 1)}


def fit_grid_baseline(gt, op, y, iters=3000, lr=0.01, l1=0.0):
    """Baseline: optimise every pixel/voxel directly with the same physics loss (and optionally the same
    sparsity term), projected to 0 <= f <= 1."""
    x = torch.zeros_like(gt, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    ynorm = (y.abs() ** 2).sum()
    t0 = time.time()
    for _ in range(iters):
        loss = ((op(x) - y).abs() ** 2).sum() / ynorm
        if l1:
            loss = loss + l1 * x.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    rec = x.detach()
    return rec, {"psnr": psnr(rec, gt), "ssim": ssim(rec, gt), "rel_l2": rel_l2(rec, gt),
                 "n_gauss": 0, "time_s": round(time.time() - t0, 1)}


def run_scatter(args, dim):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = f"scatter{dim}d"
    out = os.path.join(args.out, task)
    os.makedirs(out, exist_ok=True)
    dev = args.device
    ensure_inversebench(args.ib_dir)

    # ---- data
    if dim == 2:
        folder = ensure_lmdb_dataset("inv-scatter-test", args.data_dir, args.local_dir)
        gts = [torch.from_numpy(im).to(dev) for im in read_lmdb_images(folder, args.samples)]
        field = GaussianField(2, (128, 128), h=args.window, nonneg=True, device=dev)
        source = "InverseBench inv-scatter-test"
    else:
        if args.volumes:
            gts = load_volumes(args.volumes, args.samples, args.depth, dev, crop=args.crop)
            source = f"CytoPacq volumes from {args.volumes}"
        else:
            gts = [torch.from_numpy(synthetic_nuclei_volume(args.depth, 128, 128, seed=100 + i)).to(dev)
                   for i in range(args.samples)]
            source = "synthetic HL60-like nuclei (no --volumes given)"
        field = GaussianField(3, (args.depth, 128, 128), h=args.window, nonneg=True, device=dev)
    log(f"[{task}] {len(gts)} samples, shape {tuple(gts[0].shape)}, source: {source}")

    n_init = args.n_init or args.n_gauss // (8 if dim == 2 else 4)
    dens = dict(n_add=args.densify_add, densify_until=args.densify_until, l1=args.l1)
    rows, examples = [], {}

    # ---- A: direct fit (representation check)
    if not args.skip_direct:
        for i, gt in enumerate(gts):
            log(f"[{task}] A direct fit, sample {i}")
            p, rec, r = fit_scatter(gt, field, "direct", n_init=n_init, n_max=args.n_gauss, iters=args.iters,
                                    seed=i, log_every=max(args.iters // 4, 1))
            rows.append({"experiment": "A_direct", "receivers": None, "sample": i, "method": "gaussians", **r})
            if i == 0:
                examples["direct"] = rec.cpu()
                torch.save(p.cpu(), os.path.join(out, "A_direct_sample0_params.pt"))

    # ---- B: reconstruction from measurements
    for R in args.receivers:
        op = SliceScatter(R, dev)
        log(f"[{task}] B physics, {R} receivers (operator check vs InverseBench: {op.check(gts[0].view(-1, 128, 128)[0]):.1e})")
        for i, gt in enumerate(gts):
            gen = torch.Generator(device=dev).manual_seed(1000 + i)
            y = add_noise(op(gt), args.noise, gen)
            log(f"[{task}]   sample {i}: Gaussians")
            p, rec_g, rg = fit_scatter(gt, field, "physics", op=op, y=y, n_init=n_init, n_max=args.n_gauss,
                                       iters=args.iters, seed=i, log_every=max(args.iters // 4, 1), **dens)
            log(f"[{task}]   sample {i}: grid baseline")
            rec_b, rb = fit_grid_baseline(gt, op, y, iters=args.iters, l1=args.grid_l1)
            log(f"[{task}]   -> Gaussians {rg['psnr']:.2f} dB / SSIM {rg['ssim']:.3f}   "
                f"grid {rb['psnr']:.2f} dB / SSIM {rb['ssim']:.3f}")
            rows += [{"experiment": "B_physics", "receivers": R, "sample": i, "method": "gaussians", **rg},
                     {"experiment": "B_physics", "receivers": R, "sample": i, "method": "grid_baseline", **rb}]
            if i == 0:
                examples[R] = (rec_g.cpu(), rec_b.cpu(), rg, rb)
                torch.save(p.cpu(), os.path.join(out, f"B_physics_R{R}_sample0_params.pt"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    summ = df.groupby(["experiment", "receivers", "method"], dropna=False)[
        ["psnr", "ssim", "rel_l2", "n_gauss", "time_s"]].mean().round(3)
    summ.to_csv(os.path.join(out, "summary.csv"))
    log(f"[{task}] summary\n{summ.to_string()}")

    # ---- figure: sample 0 (middle slice for 3D)
    def sl(v):
        return v[v.shape[0] // 2] if v.ndim == 3 else v
    cols = [("ground truth", sl(gts[0].cpu()))]
    if "direct" in examples:
        cols.append(("direct fit", sl(examples["direct"])))
    for R in args.receivers:
        g, b, rg, rb = examples[R]
        cols += [(f"Gaussians {R} rec.\n{rg['psnr']:.1f} dB", sl(g)), (f"grid {R} rec.\n{rb['psnr']:.1f} dB", sl(b))]
    fig, ax = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3))
    for a, (t, im) in zip(np.atleast_1d(ax), cols):
        a.imshow(im.numpy(), cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=8)
        a.axis("off")
    fig.suptitle(f"{task}: sample 0" + (" (middle slice)" if dim == 3 else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reconstructions.png"), dpi=130)
    plt.close(fig)
    return summ
