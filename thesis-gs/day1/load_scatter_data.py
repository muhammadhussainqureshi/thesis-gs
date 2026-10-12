"""Day 1 - Step 2: load the InverseBench inverse-scattering data and run the physics once.

What it does
  1. Opens the LMDB folder from CaltechDATA (each entry = one 128x128 float32 cell image, values 0..1).
  2. Prints how many images there are and their value range.
  3. Saves a grid of the first images  -> outputs/cells_grid.png
  4. Builds the InverseBench forward model (20 transmitters, N receivers) and turns one image
     into its measurements (scattered field)  -> outputs/measurements.png
  5. Saves a few images as .npy for the next days -> outputs/cells_first8.npy

Run (from anywhere):
  python load_scatter_data.py --data ../data/inv-scatter-test --repo ../InverseBench
Options:
  --receivers 360|180|60   number of receivers (default 360)
  --n 8                    how many images to show
"""
import argparse
import os
import sys
import time

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_images(root, n):
    env = lmdb.open(root, readonly=True, lock=False, create=False)
    with env.begin(write=False) as txn:
        total = txn.stat()["entries"]
        imgs = []
        for i in range(min(n, total)):
            buf = txn.get(f"{i}".encode("utf-8"))
            imgs.append(np.frombuffer(buf, dtype=np.float32).reshape(128, 128).copy())
    env.close()
    return total, np.stack(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the inv-scatter LMDB folder (contains data.mdb)")
    ap.add_argument("--repo", required=True, help="path to the cloned InverseBench repo")
    ap.add_argument("--receivers", type=int, default=360, choices=[360, 180, 60])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- 1-3. data ----
    total, imgs = load_images(args.data, args.n)
    print(f"Entries in LMDB : {total}")
    print(f"Loaded          : {imgs.shape}  min={imgs.min():.3f}  max={imgs.max():.3f}  mean={imgs.mean():.3f}")
    np.save(os.path.join(args.out, "cells_first8.npy"), imgs)

    cols = min(4, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax, (i, im) in zip(axes.flat, enumerate(imgs)):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"sample {i}")
        ax.axis("off")
    for ax in axes.flat[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Inverse scattering: permittivity images (ground truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cells_grid.png"), dpi=120)
    print(f"Saved           : {args.out}/cells_grid.png")

    # ---- 4. physics ----
    sys.path.insert(0, os.path.abspath(args.repo))
    from inverse_problems.inverse_scatter import InverseScatter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    # Same settings as configs/problem/inv-scatter.yaml; svd=False skips the slow SVD (not needed for us).
    op = InverseScatter(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=args.receivers, numTrans=20,
                        sensorRadius=1.6, sigma_noise=1e-4, unnorm_shift=1.0, unnorm_scale=0.5,
                        svd=False, device=device)
    print(f"Forward model   : built in {time.time() - t0:.1f}s on {device}")

    # Our Gaussians will produce images in the raw 0..1 range, so we call forward(..., unnormalize=False).
    # (InverseBench feeds [-1,1] images and unnormalizes internally: (x+1)*0.5 -> same 0..1 range.)
    # NOTE: forward() only uses the first item of a batch -> always pass one image at a time.
    f = torch.from_numpy(imgs[0]).to(device)[None, None]
    y = op.forward(f, unnormalize=False)                      # (1, numTrans, numRec), complex
    y_noisy = y + op.sigma_noise * torch.randn_like(y)        # what the benchmark observes
    print(f"Measurements    : shape {tuple(y.shape)} (transmitters x receivers), dtype {y.dtype}")
    print(f"|y| range       : {y.abs().min().item():.2e} .. {y.abs().max().item():.2e}")

    # sanity check: the benchmark's normalized path gives the same measurements
    y_bench = op.forward((f - 0.5) / 0.5, unnormalize=True)
    print(f"Consistency     : max |difference| vs. benchmark path = {(y - y_bench).abs().max().item():.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(imgs[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("ground truth (sample 0)")
    axes[0].axis("off")
    for ax, data, title in [(axes[1], y_noisy[0].abs(), "|scattered field|"),
                            (axes[2], torch.angle(y_noisy[0]), "phase of scattered field")]:
        im = ax.imshow(data.cpu().numpy(), aspect="auto", cmap="viridis")
        ax.set_xlabel("receiver index")
        ax.set_ylabel("transmitter index")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"What the solver sees: {args.receivers} receivers x 20 transmitters")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "measurements.png"), dpi=120)
    print(f"Saved           : {args.out}/measurements.png")
    print("\nDay 1 data check: DONE")


if __name__ == "__main__":
    main()
