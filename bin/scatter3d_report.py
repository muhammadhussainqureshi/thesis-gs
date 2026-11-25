#!/usr/bin/env python3
"""Report for the 3D controls (advisor, Nov 21): true 3D Born, BBBC024 test volumes, 60 receivers.

    python bin/scatter3d_report.py RESULTS_DIR [RESULTS_DIR ...] --out scatter3d_report [--ref2d CSV]

Run folders: s<SNR>_tune_tv, s<SNR>_nonet, s<SNR>_pointmamba, s34_gm_mlp, s34_gm_random, steps2d_nonet_6000.
Per volume, Gaussian results are averaged over seeds; paired differences use a t-interval over volumes.
--ref2d: results.csv of the 3000-iteration 2D run with the same settings (for the extra-steps check).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

LABEL = {"nonet": "Gaussians (B+G), no network", "pointmamba": "+ PointMamba", "gm_mlp": "+ per-Gaussian MLP",
         "gm_random": "+ Mamba, random order"}


def md(df):
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def t_ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def load(dirs, task):
    out = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", task, "results.csv"), recursive=True):
            run = os.path.relpath(f, d).split(os.sep)[0]
            df = pd.read_csv(f)
            df["run"] = run
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="scatter3d_report")
    ap.add_argument("--ref2d", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.dirs, "scatter3d")
    lines = ["# 3D controls: true 3D Born, BBBC024, 60 receivers\n",
             "Test volumes 2-11 (volumes 0-1 only for tuning the TV weight). Noise: complex Gaussian on the "
             "measured field, sigma = 2% (SNR 34 dB) or 10% (SNR 20 dB) of rms(y).\n"]
    if df.empty:
        print("no scatter3d results found")
        return
    df["snr"] = df["run"].str.extract(r"^s(\d+)_")[0]
    df["model"] = df["run"].str.replace(r"^s\d+_", "", regex=True)

    for snr, d in df.groupby("snr"):
        tune = d[d["model"] == "tune_tv"]
        if not tune.empty:
            t = tune[tune["method"].str.startswith("grid_tv")].groupby("tv")["psnr"].mean().round(2)
            g = tune[tune["method"] == "grid_baseline"]["psnr"].mean()
            lines += [f"## SNR {snr} dB: TV weight on tuning volumes 0-1 (PSNR dB; sparsity-only grid {g:.2f})\n",
                      md(t.to_frame("PSNR")), f"\nChosen: {t.idxmax()}\n"]
        test = d[d["model"] != "tune_tv"]
        per_vol = {}
        for model, g in test[test["method"] == "gaussians"].groupby("model"):
            per_vol[LABEL.get(model, model)] = g.groupby("sample")[["psnr", "ssim"]].mean()
        grids = test[test["method"] != "gaussians"]
        for m, g in grids.groupby("method"):
            name = "Voxel grid + sparsity" if m == "grid_baseline" else f"Voxel grid + sparsity + TV ({m[7:]})"
            per_vol[name] = g.groupby("sample")[["psnr", "ssim"]].mean()
        rows = []
        for name, v in per_vol.items():
            seeds = test[(test["model"].map(lambda x: LABEL.get(x, x)) == name) & (test["method"] == "gaussians")]["seed"].nunique()
            rows.append({"method": name, "volumes": len(v), "seeds": seeds or 1, "PSNR": v["psnr"].mean(),
                         "PSNR SD (volumes)": v["psnr"].std(ddof=1), "SSIM": v["ssim"].mean()})
        tab = pd.DataFrame(rows).set_index("method").round(3)
        tab.to_csv(os.path.join(a.out, f"snr{snr}_table.csv"))
        lines += [f"## SNR {snr} dB: test volumes\n", md(tab), "\n"]
        ref = LABEL["nonet"]
        if ref in per_vol:
            pr = []
            for name, v in per_vol.items():
                if name == ref:
                    continue
                if name.startswith("+"):          # network on top of the Gaussians: network minus no network
                    label, dv = f"{name[2:]} - no network", v["psnr"] - per_vol[ref]["psnr"]
                else:                             # grid baseline: Gaussians minus grid
                    label, dv = f"Gaussians - {name}", per_vol[ref]["psnr"] - v["psnr"]
                dv = dv.dropna()
                m, lo, hi = t_ci(dv)
                pr.append({"comparison": label, "mean dB": round(m, 2), "95% CI": f"[{lo:+.2f}, {hi:+.2f}]",
                           "first better": f"{int((dv > 0).sum())}/{len(dv)}"})
            p = pd.DataFrame(pr).set_index("comparison")
            lines += [f"Paired differences (per volume, seed-averaged):\n",
                      md(p), "\n"]

    d2 = load(a.dirs, "scatter2d")
    if not d2.empty and a.ref2d and os.path.exists(a.ref2d):
        ref = pd.read_csv(a.ref2d)
        ref = ref[ref["method"] == "gaussians"]
        new = d2[d2["method"] == "gaussians"]
        rows = []
        for R in sorted(new["receivers"].unique(), reverse=True):
            x = ref[ref["receivers"] == R].set_index("sample")["psnr"]
            y = new[new["receivers"] == R].set_index("sample")["psnr"]
            m, lo, hi = t_ci((y - x).dropna())
            rows.append({"receivers": R, "3000 it": x.mean(), "6000 it": y.mean(), "difference": m,
                         "95% CI": f"[{lo:+.2f}, {hi:+.2f}]"})
        t = pd.DataFrame(rows).set_index("receivers").round(2)
        lines += ["## Extra optimisation steps (2D, 10 cells, no network): 3000 vs 6000 iterations\n", md(t), "\n"]

    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
