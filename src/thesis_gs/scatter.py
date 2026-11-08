"""Inverse scattering with Gaussians (2D and 3D): direct fit, physics fit, pixel/voxel baseline, experiment runner."""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import GaussianField
from .metrics import psnr, rel_l2, ssim
from .physics import Born3D, SliceScatter, add_noise
from .utils import cosine_lr, ensure_inversebench, ensure_lmdb_dataset, log, read_lmdb_images
from .volumes import load_volumes, synthetic_nuclei_volume


def saturate(s):
    """Bounded density f = 1 - exp(-s) in [0, 1): overlapping Gaussians cannot overshoot (like opacity in 3DGS)."""
    return 1.0 - torch.exp(-s)


def fit_scatter(gt, field, mode, op=None, y=None, n_init=400, n_max=4000, iters=3000, lr=0.05,
                densify_every=250, n_add=None, densify_until=0.6, l1=0.0, seed=0, log_every=500,
                bounded=False, init_field=None):
    """mode='direct': MSE to ground truth.  mode='physics': ||A f - y||^2/||y||^2 (gt only for metrics).
    bounded: the field is 1 - exp(-sum of Gaussians), so it stays in [0, 1) like the clipped grid baseline.
    init_field: start from Gaussians fitted to this field (e.g. the grid reconstruction) instead of uniform."""
    g = torch.Generator(device=gt.device).manual_seed(seed)
    n_add = n_add or max(50, n_init // 4)
    if mode == "direct":
        p = field.init_from_field(gt, n_init, generator=g)
    elif init_field is not None:
        src = -torch.log1p(-init_field.clamp(0, 0.95)) if bounded else init_field
        p = field.init_from_field(src, n_init, generator=g)
    else:
        p = field.init_uniform(n_init, generator=g)
    p = p.requires_grad_(True)
    render = (lambda q: saturate(field.render(q))) if bounded else field.render
    opt = torch.optim.Adam([p], lr=lr)
    ynorm = (y.abs() ** 2).sum() if y is not None else None
    t0 = time.time()
    for it in range(1, iters + 1):
        img = render(p)
        loss = F.mse_loss(img, gt) if mode == "direct" else ((op(img) - y).abs() ** 2).sum() / ynorm
        if l1 and mode != "direct":
            loss = loss + l1 * img.mean()          # sparsity: no mass where the measurements don't ask for it
        opt.zero_grad()
        loss.backward()
        opt.step()
        cosine_lr(opt, lr, it, iters)
        if densify_every and it % densify_every == 0 and it <= densify_until * iters:
            with torch.no_grad():
                img = render(p)
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
                log(f"    iter {it:5d}  loss {loss.item():.3e}  PSNR {psnr(render(p).clamp(0, 1), gt):6.2f} dB  #G {p.shape[0]}")
    with torch.no_grad():
        rec = render(p).clamp(0, 1)
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
        field = GaussianField(2, (128, 128), h=args.window, nonneg=True, device=dev, beta=args.gauss_beta)
        source = "InverseBench inv-scatter-test"
    else:
        if args.volumes:
            gts = load_volumes(args.volumes, args.samples, args.depth, dev, crop=args.crop)
            source = f"CytoPacq volumes from {args.volumes}"
        else:
            gts = [torch.from_numpy(synthetic_nuclei_volume(args.depth, 128, 128, seed=100 + i)).to(dev)
                   for i in range(args.samples)]
            source = "synthetic HL60-like nuclei (no --volumes given)"
        field = GaussianField(3, (args.depth, 128, 128), h=args.window, nonneg=True, device=dev, beta=args.gauss_beta)
    born = dim == 3 and args.physics == "born3d"
    if born and args.noise_rel is None:
        args.noise_rel = 0.02          # ~ the SNR of the InverseBench 2D setting (sigma 1e-4 vs rms(y) ~ 5e-3)
    log(f"[{task}] {len(gts)} samples, shape {tuple(gts[0].shape)}, source: {source}, "
        f"physics: {'true 3D Born, %d transmitters' % args.transmitters if born else 'InverseBench 2D operator' + (' per slice' if dim == 3 else '')}, "
        f"kernel beta={args.gauss_beta}")

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
        if born:
            op = Born3D(tuple(gts[0].shape), R, dev, transmitters=args.transmitters)
            log(f"[{task}] B physics, {R} receivers, {op.n_meas} measurements for {gts[0].numel()} voxels "
                f"(adjoint check: {op.check(gts[0]):.1e})")
        else:
            op = SliceScatter(R, dev)
            log(f"[{task}] B physics, {R} receivers (operator check vs InverseBench: {op.check(gts[0].view(-1, 128, 128)[gts[0].view(-1, 128 * 128).sum(1).argmax()]):.1e})")
        for i, gt in enumerate(gts):
            gen = torch.Generator(device=dev).manual_seed(1000 + i)
            y0 = op(gt)
            sigma = args.noise_rel * y0.abs().pow(2).mean().sqrt().item() if args.noise_rel else args.noise
            y = add_noise(y0, sigma, gen)
            log(f"[{task}]   sample {i}: Gaussians")
            init_field, grid_done = None, None
            if args.init_from_grid:
                log(f"[{task}]   sample {i}: grid solution (baseline and initialisation)")
                grid_done = fit_grid_baseline(gt, op, y, iters=args.iters, l1=args.grid_l1)
                init_field = grid_done[0]
            p, rec_g, rg = fit_scatter(gt, field, "physics", op=op, y=y, n_init=n_init, n_max=args.n_gauss,
                                       iters=args.iters, seed=i, log_every=max(args.iters // 4, 1),
                                       bounded=args.bounded, init_field=init_field, **dens)
            rg["n_params"] = rg["n_gauss"] * field.P
            if grid_done:
                rg["time_s"] = round(rg["time_s"] + grid_done[1]["time_s"], 1)   # include the initialising grid fit
            rows.append({"experiment": "B_physics", "receivers": R, "sample": i, "method": "gaussians", **rg})
            if args.skip_grid:
                rec_b, rb = None, None
                log(f"[{task}]   -> Gaussians {rg['psnr']:.2f} dB / SSIM {rg['ssim']:.3f}")
            else:
                log(f"[{task}]   sample {i}: grid baseline")
                rec_b, rb = grid_done or fit_grid_baseline(gt, op, y, iters=args.iters, l1=args.grid_l1)
                rb["n_params"] = gt.numel()
                log(f"[{task}]   -> Gaussians {rg['psnr']:.2f} dB / SSIM {rg['ssim']:.3f}   "
                    f"grid {rb['psnr']:.2f} dB / SSIM {rb['ssim']:.3f}")
                rows.append({"experiment": "B_physics", "receivers": R, "sample": i, "method": "grid_baseline", **rb})
            if i == 0:
                examples[R] = (rec_g.cpu(), None if rec_b is None else rec_b.cpu(), rg, rb)
                torch.save(p.cpu(), os.path.join(out, f"B_physics_R{R}_sample0_params.pt"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    cols = [c for c in ["psnr", "ssim", "rel_l2", "n_gauss", "n_params", "time_s"] if c in df]
    grp = df.groupby(["experiment", "receivers", "method"], dropna=False)
    summ = grp[cols].mean()
    std = grp[["psnr", "ssim"]].std().add_suffix("_std")      # spread over samples
    summ = summ.join(std)
    summ["n_samples"] = grp.size()
    summ = summ.round(3)
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
        cols.append((f"Gaussians {R} rec.\n{rg['psnr']:.1f} dB", sl(g)))
        if b is not None:
            cols.append((f"grid {R} rec.\n{rb['psnr']:.1f} dB", sl(b)))
    fig, ax = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3))
    for a, (t, im) in zip(np.atleast_1d(ax), cols):
        a.imshow(im.numpy(), cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=8)
        a.axis("off")
    fig.suptitle(f"{task}: sample 0" + (" (middle slice)" if dim == 3 else "")
                 + (", true 3D Born physics" if born else "") + (f", kernel beta={args.gauss_beta:g}" if args.gauss_beta != 1 else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reconstructions.png"), dpi=130)
    plt.close(fig)
    return summ
