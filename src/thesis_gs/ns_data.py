"""Navier-Stokes vorticity sequences rolled out with the InverseBench solver."""
import math
import os
import time

import torch

from .utils import ensure_lmdb_dataset, log, read_lmdb_images


def ns_sequences(args, dev):
    """Initial vorticity from InverseBench NS test set, rolled out with InverseBench's solver.
    Returns list of tensors (T, 128, 128), normalised by 10 (vorticity range is about +-10)."""
    cache = os.path.join(args.out, "ns4d", f"sequences_n{args.samples}_T{args.frames}_dt{args.frame_dt}.pt")
    if os.path.exists(cache):
        log(f"[ns4d] loading cached sequences {cache}")
        return [s.to(dev) for s in torch.load(cache)]
    from inverse_problems.navier_stokes import NavierStokes2d
    folder = ensure_lmdb_dataset("navier-stokes-test", args.data_dir, args.local_dir)
    w0s = read_lmdb_images(folder, args.samples)
    solver = NavierStokes2d(128, 128, 2 * math.pi, 2 * math.pi, device=dev, dtype=torch.float64)
    t = torch.linspace(0, 2 * math.pi, 129, device=dev, dtype=torch.float64)[:-1]
    _, yy = torch.meshgrid(t, t, indexing="ij")
    force = -4 * torch.cos(4.0 * yy)
    seqs = []
    for i, w0 in enumerate(w0s):
        t0 = time.time()
        w = torch.from_numpy(w0).to(dev, torch.float64)[None]
        frames = [w[0].float()]
        for _ in range(args.frames - 1):
            w = solver.solve(w, force, args.frame_dt, 200.0, adaptive=True)
            frames.append(w[0].float())
        seqs.append(torch.stack(frames) / 10.0)
        log(f"[ns4d] rolled out sample {i}: {args.frames} frames x dt={args.frame_dt} in {time.time() - t0:.1f}s")
    torch.save([s.cpu() for s in seqs], cache)
    return seqs
