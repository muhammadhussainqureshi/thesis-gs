#!/usr/bin/env python3
"""Step-1 report for the 4D flow experiment: does Mamba beat the MLP and the GRU when all get the same smoothness
term (weight tuned per model on validation frames), matched network size and the same training budget?

    python bin/ns4d_step1_report.py RESULTS_DIR [RESULTS_DIR ...] --out step1_report

RESULTS_DIR = output folders of the step-1 notebooks (run folders named tune_*, final_*, baseline_*, ceiling_*).
Criterion (fixed before seeing results): Mamba wins if its mean test PSNR over 5 flows x 3 seeds is >= 1 dB
higher than both the MLP and the GRU, and the 95% CI of each paired difference (per flow, seed-averaged,
t-interval) lies above 0. Writes report.md, per_frame.png, spectra.png and the tables as CSV.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

MARGIN = 1.0
MODELS = ["mlp", "gru", "mamba"]


def load(dirs, fname):
    dfs = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "ns4d", fname), recursive=True):
            run = os.path.relpath(f, d).split(os.sep)[0]
            df = pd.read_csv(f)
            df["run"] = run
            df["stage"] = run.split("_")[0]
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def md_table(df):
    try:
        return df.to_markdown()
    except ImportError:                                        # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


def t_ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="step1_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, pf, sp = load(a.dirs, "results.csv"), load(a.dirs, "per_frame.csv"), load(a.dirs, "spectra.csv")
    if res.empty:
        raise SystemExit("no results.csv found under " + " ".join(a.dirs))
    md = ["# Step 1: Mamba vs MLP vs GRU with the same smoothness term\n",
          f"Criterion (fixed in advance): Mamba wins if its test PSNR is >= {MARGIN} dB higher than both the MLP and "
          "the GRU (mean over 5 flows x 3 seeds) and the 95% CI of each paired difference is above 0.\n"]

    tune = res[res["stage"] == "tune"]
    if not tune.empty:
        t = tune.pivot_table(index="temporal_reg", columns="deform", values="val_frames_psnr", aggfunc="mean")
        t.to_csv(os.path.join(a.out, "tuning.csv"))
        md += ["## Smoothness weight, chosen on validation frames (mean validation PSNR, dB)\n",
               md_table(t.round(2)), "\n", "Chosen: " + ", ".join(f"{m} λ={t[m].idxmax()}" for m in t.columns), "\n"]

    fin = res[(res["stage"].isin(["final", "baseline"])) & (res["obs_ratio"] < 1)]
    fin_m = fin[fin["deform"].isin(MODELS)]
    if fin_m.empty:
        md.append("No final runs found yet.\n")
    else:
        rows = []
        for kind, g in fin.groupby("deform"):
            per_flow = g.groupby("sample")["heldout_frames_psnr"].mean()
            seed_sd = g.groupby("sample")["heldout_frames_psnr"].std().mean()
            rows.append({"model": kind, "λ": g["temporal_reg"].iloc[0], "net params": int(g["net_params"].iloc[0]),
                         "runs": len(g), "test PSNR": g["heldout_frames_psnr"].mean(),
                         "SD across flows": per_flow.std(ddof=1), "SD across seeds": seed_sd,
                         "observed px": g["train_obs_psnr"].mean(), "unobserved px": g["train_unobs_psnr"].mean(),
                         "spectrum log err": g["spec_log_err"].mean(),
                         "enstrophy ratio": g["enstrophy_ratio_test"].mean(), "min/run": g["time_s"].mean() / 60})
        tab = pd.DataFrame(rows).set_index("model")
        tab.to_csv(os.path.join(a.out, "final_table.csv"))
        md += ["## Final runs, 10% observed (PSNR dB; test = 5 never-observed frames)\n", md_table(tab.round(3)), "\n"]
        if "mamba" in tab.index:
            pm = tab.loc["mamba", "net params"]
            md.append("Parameter match vs Mamba: " + ", ".join(
                f"{m} {tab.loc[m, 'net params'] / pm - 1:+.1%}" for m in ("mlp", "gru") if m in tab.index) +
                " (tolerance ±10%)\n")

        verdict, lines = True, []
        flow_mean = fin_m.groupby(["deform", "sample"])["heldout_frames_psnr"].mean().unstack(0)
        for other in ("mlp", "gru"):
            if "mamba" not in flow_mean or other not in flow_mean:
                verdict = None
                lines.append(f"- Mamba − {other.upper()}: missing runs")
                continue
            d = (flow_mean["mamba"] - flow_mean[other]).dropna()
            m, lo, hi = t_ci(d)
            ok = bool(m >= MARGIN and lo > 0)
            verdict = (verdict and ok) if verdict is not None else None
            lines.append(f"- Mamba − {other.upper()}: {m:+.2f} dB, 95% CI [{lo:+.2f}, {hi:+.2f}] over {len(d)} flows "
                         f"(per flow: {', '.join(f'{v:+.2f}' for v in d)}) → {'meets' if ok else 'does not meet'} "
                         "the criterion")
        n_seeds = fin_m.groupby("deform")["seed"].nunique().to_dict()
        md += ["## Paired differences (seed-averaged per flow)\n"] + lines + [
            f"\nSeeds per model: {n_seeds}\n",
            "**Verdict: " + ("Mamba wins.**" if verdict else "Mamba does not win.**" if verdict is False
                             else "incomplete (missing runs).**") + "\n"]

    ceil = res[res["obs_ratio"] >= 1]
    if not ceil.empty:
        c = ceil.groupby("deform")[["heldout_frames_psnr", "train_frames_psnr"]].mean()
        md += ["## Ceiling: fully observed (100% of pixels in training frames)\n", md_table(c.round(2)), "\n"]

    if not pf.empty:
        p = pf[(pf["stage"].isin(["final", "baseline"])) & (pf["obs_ratio"] < 1)]
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (kind, g) in enumerate(p.groupby("deform")):
            g = g.groupby("frame").agg(all=("psnr_all", "mean"), obs=("psnr_obs", "mean"), unobs=("psnr_unobs", "mean"),
                                       ftype=("frame_type", "first"))
            col = f"C{i}"
            if kind != "interp":                               # interpolation reproduces observed pixels exactly
                ax.plot(g.index, g["obs"], ":", color=col, marker=".", label=f"{kind}: observed px (fit)")
            ax.plot(g.index, g["unobs"], "--", color=col, marker=".", label=f"{kind}: unobserved px")
            te = g[g["ftype"] == "test"]
            ax.plot(te.index, te["all"], "o", color=col, ms=8, label=f"{kind}: test frames")
        ax.set_ylim(top=min(ax.get_ylim()[1], 60))
        ax.set_xlabel("frame")
        ax.set_ylabel("PSNR (dB)")
        ax.set_xticks(range(0, int(p["frame"].max()) + 1, 2))
        ax.legend(fontsize=6, ncol=3)
        ax.set_title("Error split per frame, 10% observed (mean over flows and seeds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "per_frame.png"), dpi=150)
        plt.close(fig)
        md.append("![per-frame error split](per_frame.png)\n")
    if not sp.empty:
        s = sp[(sp["stage"].isin(["final", "baseline"])) & (sp["obs_ratio"] < 1)]
        fig, ax = plt.subplots(figsize=(5, 4))
        gt = s.groupby("k")["E_gt"].mean()
        ax.loglog(gt.index, gt.values, "k-", lw=2, label="ground truth")
        for kind, g in s.groupby("deform"):
            e = g.groupby("k")["E_rec"].mean()
            ax.loglog(e.index, e.values, label=kind)
        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("E(k), test frames")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "spectra.png"), dpi=150)
        plt.close(fig)
        md.append("![energy spectra](spectra.png)\n")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
