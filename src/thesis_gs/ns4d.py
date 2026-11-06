"""4D Gaussian Splatting on Navier-Stokes: canonical Gaussians + deformation, training and evaluation."""
import os
import time

import numpy as np
import torch

from .deform import DeformMLP, DeformMamba, StaticDeform, apply_deform
from .gaussians import GaussianField
from .metrics import psnr, rel_l2, ssim
from .ns_data import ns_sequences
from .utils import cosine_lr, ensure_inversebench, log


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
    test_idx = torch.arange(2, T, 4, device=dev)                       # held-out time steps
    train_idx = torch.tensor([k for k in range(T) if k not in set(test_idx.tolist())], device=dev)
    field = GaussianField(2, (128, 128), h=args.window, nonneg=False, wrap=True, device=dev)  # periodic domain
    log(f"[ns4d] {len(seqs)} sequences x {T} frames; train frames {len(train_idx)}, held-out {len(test_idx)}; "
        f"observed pixel ratios {args.obs_ratios}; deformations {args.deform}")

    rows, examples = [], {}
    for ratio in args.obs_ratios:
        for si, seq in enumerate(seqs):
            gen = torch.Generator(device=dev).manual_seed(500 + si)
            mask = (torch.rand(seq.shape, generator=gen, device=dev) < ratio).float()   # observed pixels per frame
            obs = seq + args.ns_noise * torch.randn(seq.shape, generator=gen, device=dev)
            for kind in args.deform:
                torch.manual_seed(si)
                net = {"mlp": DeformMLP, "mamba": DeformMamba, "static": StaticDeform,
                       "mamba_bi": lambda: DeformMamba(bidirectional=True)}[kind]().to(dev)
                g0 = torch.Generator(device=dev).manual_seed(si)
                canon = field.init_from_field(obs[0] * mask[0], args.n_gauss, scale=2.0, generator=g0)
                canon.requires_grad_(True)
                groups = [{"params": [canon], "lr": args.lr_gauss, "base_lr": args.lr_gauss}]
                if any(True for _ in net.parameters()):
                    groups.append({"params": list(net.parameters()), "lr": args.lr_net, "base_lr": args.lr_net})
                opt = torch.optim.Adam(groups)
                t0 = time.time()
                log(f"[ns4d] ratio {ratio}, sample {si}, deform={kind}")
                for it in range(1, args.iters + 1):
                    bidx = train_idx[torch.randint(len(train_idx), (min(args.batch_frames, len(train_idx)),),
                                                   device=dev)]
                    delta = net(canon[:, :2] / 128.0, t_all, bidx)
                    params = apply_deform(canon, delta)
                    pred = torch.stack([field.render(params[b]) for b in range(len(bidx))])
                    m = mask[bidx]
                    loss = (((pred - obs[bidx]) ** 2) * m).sum() / m.sum().clamp(min=1)
                    if args.temporal_reg and kind != "static":
                        # smooth motion: offsets at t and t+1 should be close (also covers unobserved frames)
                        delta_next = net(canon[:, :2] / 128.0, t_all, (bidx + 1).clamp(max=T - 1))
                        loss = loss + args.temporal_reg * ((delta_next - delta) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    cosine_lr(opt, None, it, args.iters)
                    if it % max(args.iters // 4, 1) == 0 or it == args.iters:
                        log(f"    iter {it:5d}  loss {loss.item():.3e}")
                train_time = time.time() - t0
                with torch.no_grad():
                    allp = apply_deform(canon, net(canon[:, :2] / 128.0, t_all, torch.arange(T, device=dev)))
                    rec = torch.stack([field.render(allp[k]) for k in range(T)])
                res = {}
                for name, idx in [("train_frames", train_idx), ("heldout_frames", test_idx)]:
                    res[f"{name}_rel_l2"] = float(np.mean([rel_l2(rec[k], seq[k]) for k in idx.tolist()]))
                    res[f"{name}_psnr"] = float(np.mean([psnr(rec[k], seq[k], 2.0) for k in idx.tolist()]))
                    res[f"{name}_ssim"] = float(np.mean([ssim(rec[k], seq[k], 2.0) for k in idx.tolist()]))
                n_net = sum(p.numel() for p in net.parameters())
                rows.append({"obs_ratio": ratio, "sample": si, "deform": kind, **res, "n_gauss": canon.shape[0],
                             "net_params": n_net, "time_s": round(train_time, 1)})
                log(f"    -> rel L2 train {res['train_frames_rel_l2']:.4f} | held-out {res['heldout_frames_rel_l2']:.4f}"
                    f"   PSNR held-out {res['heldout_frames_psnr']:.2f} dB  ({train_time:.0f}s)")
                if si == 0:
                    examples[(ratio, kind)] = rec.cpu()
                    torch.save({"canonical": canon.detach().cpu(), "net": net.state_dict()},
                               os.path.join(out, f"model_ratio{ratio}_{kind}_sample0.pt"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    cols = [c for c in df.columns if c.endswith(("rel_l2", "psnr", "ssim"))] + ["time_s"]
    summ = df.groupby(["obs_ratio", "deform"])[cols].mean().round(4)
    summ.to_csv(os.path.join(out, "summary.csv"))
    log(f"[ns4d] summary\n{summ.to_string()}")

    # figure: sample 0, a few frames incl. held-out ones, for every (ratio, deform)
    show = sorted(set([0, int(test_idx[0]), T // 2, int(test_idx[-1]), T - 1]))
    keys = list(examples.keys())
    fig, ax = plt.subplots(1 + len(keys), len(show), figsize=(2.2 * len(show), 2.2 * (1 + len(keys))), squeeze=False)
    for j, k in enumerate(show):
        tag = " (held-out)" if k in set(test_idx.tolist()) else ""
        ax[0, j].imshow(seqs[0][k].cpu(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0, j].set_title(f"GT t={k}{tag}", fontsize=7)
        for i, key in enumerate(keys):
            ax[i + 1, j].imshow(examples[key][k], cmap="RdBu_r", vmin=-1, vmax=1)
            ax[i + 1, j].set_title(f"{key[1]}, obs {key[0]:.0%}", fontsize=7)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reconstructions.png"), dpi=130)
    plt.close(fig)
    return summ
