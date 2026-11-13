"""thesis-gs command line: Gaussian Splatting for physical inverse problems (InverseBench data).

Tasks:
  scatter2d : 2D Gaussians reconstruct InverseBench cell images from scattered-field measurements.
  scatter3d : 3D Gaussians reconstruct 3D cell volumes, with the 2D operator per slice (--physics slice)
              or true 3D Born scattering with sensors on a sphere (--physics born3d).
  ns4d      : 4D Gaussian Splatting (2D + time) on Navier-Stokes vorticity; deformation MLP / Mamba / static.

Usage:  python -m thesis_gs --task all --data_dir data --out outputs      (add --quick for a smoke test)
"""
import argparse
import json
import os
import time

import torch

from .ns4d import run_ns4d
from .scatter import run_scatter
from .utils import log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", choices=["scatter2d", "scatter3d", "ns4d", "all"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data_dir", default="data", help="where the CaltechDATA zips are (or will be downloaded)")
    ap.add_argument("--local_dir", default="/tmp/thesis_data", help="fast local folder for unzipped data")
    ap.add_argument("--ib_dir", default="/tmp/InverseBench", help="InverseBench clone (cloned if missing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--n_gauss", type=int, default=None, help="max Gaussians (default 2000 2D / 8000 3D / 2000 4D)")
    ap.add_argument("--window", type=int, default=None, help="local window half-size (default 8 / 6 / 10)")
    # scattering
    ap.add_argument("--receivers", type=int, nargs="+", default=[360, 180, 60])
    ap.add_argument("--noise", type=float, default=1e-4, help="measurement noise (benchmark: 1e-4)")
    ap.add_argument("--volumes", default=None, help="folder with CytoPacq .tif/.npy volumes for scatter3d")
    ap.add_argument("--depth", type=int, default=32, help="number of z-slices for 3D volumes")
    ap.add_argument("--skip_direct", action="store_true")
    ap.add_argument("--crop", type=float, default=0.5, help="central Y/X fraction kept from loaded volumes")
    ap.add_argument("--n_init", type=int, default=None, help="initial Gaussians (default n_gauss/8 2D, /4 3D)")
    ap.add_argument("--densify_add", type=int, default=None, help="Gaussians added per densify step (default n_init/4)")
    ap.add_argument("--densify_until", type=float, default=0.6, help="densify during this fraction of iterations")
    ap.add_argument("--l1", type=float, default=0.0, help="sparsity weight on the mean of the Gaussian field (physics fit)")
    ap.add_argument("--physics", default="slice", choices=["slice", "born3d"],
                    help="scatter3d forward model: InverseBench 2D operator per z-slice, or true 3D Born scattering")
    ap.add_argument("--transmitters", type=int, default=60, help="transmitters on the sphere (born3d only)")
    ap.add_argument("--noise_rel", type=float, default=None,
                    help="noise std relative to rms(y); default 0.02 for born3d (~ InverseBench 2D SNR), else --noise is used")
    ap.add_argument("--gauss_beta", type=float, default=1.0,
                    help="kernel exponent: 1 = Gaussian, >1 = generalised Gaussian with sharper edges")
    ap.add_argument("--bounded", action="store_true",
                    help="bounded density f = 1 - exp(-sum of Gaussians), in [0, 1) like the clipped grid")
    ap.add_argument("--init_from_grid", action="store_true",
                    help="initialise Gaussians from the grid reconstruction (Gaussians then compress and refine it)")
    ap.add_argument("--gmamba", default="none", choices=["none", "mlp", "random", "hilbert", "pointmamba"],
                    help="PointMamba over the Gaussians (per scene): network type / serialisation of the patches")
    ap.add_argument("--gm_residual", action="store_true",
                    help="give the PointMamba tokens the back-projected measurement residual at each Gaussian")
    ap.add_argument("--skip_grid", action="store_true", help="skip the pixel/voxel baseline (e.g. when already run)")
    ap.add_argument("--grid_l1", type=float, default=0.0, help="same sparsity weight for the pixel/voxel baseline")
    # Navier-Stokes 4D
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--frame_dt", type=float, default=0.05)
    ap.add_argument("--obs_ratios", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--ns_noise", type=float, default=0.0)
    ap.add_argument("--deform", nargs="+", default=["mlp", "mamba"], choices=["mlp", "mamba", "mamba_bi", "mamba_st", "mamba_vel", "mamba_st_vel", "static"])
    ap.add_argument("--batch_frames", type=int, default=4)
    ap.add_argument("--temporal_reg", type=float, default=0.0, help="weight of the deformation smoothness term over time")
    ap.add_argument("--lr_gauss", type=float, default=0.01)
    ap.add_argument("--lr_net", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()

    if args.quick:
        args.samples, args.iters = 1, 40
        args.receivers, args.depth, args.frames = [360, 60], 16, 6
        args.obs_ratios = [0.1]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log(f"device={args.device}  task={args.task}  out={args.out}")
    if args.device == "cpu" and not args.quick:
        log("WARNING: running on CPU will be very slow. In Colab: Runtime -> Change runtime type -> T4 GPU.")

    tasks = ["scatter2d", "scatter3d", "ns4d"] if args.task == "all" else [args.task]
    user_n, user_w = args.n_gauss, args.window
    for task in tasks:
        defaults = {"scatter2d": (2000, 8), "scatter3d": (8000, 6), "ns4d": (2000, 10)}[task]
        args.n_gauss = user_n or (defaults[0] // (10 if args.quick else 1))
        args.window = user_w or defaults[1]
        t0 = time.time()
        if task == "scatter2d":
            run_scatter(args, 2)
        elif task == "scatter3d":
            run_scatter(args, 3)
        else:
            run_ns4d(args)
        log(f"===== {task} finished in {(time.time() - t0) / 60:.1f} min. Results in {os.path.join(args.out, task)}")



if __name__ == "__main__":
    main()
