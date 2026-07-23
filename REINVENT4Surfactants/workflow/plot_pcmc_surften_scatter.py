#!/usr/bin/env python3
"""
Scatter plot of pCMC vs. surface tension at reference concentration
(`SurfTen` = `surface_tension_avg`, MD-derived -- see README for why this is
NOT `AW_ST_CMC`, the experimental "surface tension at CMC", and why its units
aren't confirmed to be literal mN/m), split into two panels: default vs.
optimized hyperparameters. Each panel shows its own generated molecules, the
SurfPro training set, a ZINC sample, the ZINC "best of the best" tail, and the
SurfPro holdout split by whether that panel's run rediscovered it.

REINVENT's own "(raw)" score columns are normalized/inverted [0,1] component
scores, not real units -- un-normalized here using config.json's calibration
bounds. pCMC is maximized (minimize=False, direct: actual = min + score*range)
as of the 2026-07-22 direction fix; SurfTen is minimized (inverted: actual =
min + (1-score)*range).

Saved locally only -- not published anywhere.

Usage:
    python workflow/plot_pcmc_surften_scatter.py --out pcmc_surften_scatter.png
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "workflow")
from evaluate_run import canon

PCMC_MIN, PCMC_MAX = 0.0089955596692448, 6.79588001734408
SURFTEN_MIN, SURFTEN_MAX = 173.98984, 594.85364


def unnorm_pcmc(score):
    return PCMC_MIN + score * (PCMC_MAX - PCMC_MIN)


def unnorm_surften(score):
    return SURFTEN_MIN + (1 - score) * (SURFTEN_MAX - SURFTEN_MIN)


def load_generated(glob_pattern):
    paths = sorted(glob.glob(glob_pattern))
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    props = pd.DataFrame({
        "pCMC": unnorm_pcmc(raw["pCMC (raw)"]),
        "SurfTen": unnorm_surften(raw["SurfTen (raw)"]),
        "canon": raw["SMILES"].apply(canon),
    })
    return props


def plot_panel(ax, title, gen_full, gen_sample_n, train, holdout, zinc, zinc_best, seed):
    rng = np.random.RandomState(seed)
    gen = gen_full.sample(n=min(gen_sample_n, len(gen_full)), random_state=rng)

    hit_set = set(gen_full["canon"].dropna())
    rediscovered = holdout[holdout["canon"].isin(hit_set)]
    missed = holdout[~holdout["canon"].isin(hit_set)]

    ax.scatter(gen["pCMC"], gen["SurfTen"], s=16, alpha=0.5, color="crimson",
               label=f"Generated (n={gen_sample_n} of {len(gen_full)})", zorder=1)
    ax.scatter(train["pCMC"], train["SurfTen"], s=16, alpha=0.6, color="steelblue",
               label=f"SurfPro training set (n={len(train)})", zorder=2)
    ax.scatter(zinc["pCMC"], zinc["SurfTen"], s=8, alpha=0.5, color="forestgreen",
               marker="^", label=f"ZINC sample (n={len(zinc)})", zorder=3)
    ax.scatter(zinc_best["pCMC"], zinc_best["SurfTen"], s=22, alpha=0.85, color="darkgreen",
               marker="^", label=f"ZINC best-of-best (n={len(zinc_best)})", zorder=3)
    ax.scatter(missed["pCMC"], missed["SurfTen"], s=55, alpha=0.9, color="black",
               marker="x", linewidths=2, label=f"SurfPro holdout, missed (n={len(missed)})", zorder=4)
    ax.scatter(rediscovered["pCMC"], rediscovered["SurfTen"], s=60, alpha=0.95,
               facecolors="gold", edgecolors="black", linewidths=0.8,
               marker="D", label=f"SurfPro holdout, rediscovered (n={len(rediscovered)})", zorder=5)

    ax.set_xlabel("pCMC  (higher = lower CMC = more efficient)")
    ax.set_ylabel("Surface tension at reference concentration  (lower = better)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(0, 650)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--holdout-csv", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--zinc-scored", default="data/ZINC/zinc_scored_9props.csv.gz")
    ap.add_argument("--zinc-best-csv", default="data/zinc_holdout_low_pCMC_low_SurfTen.csv")
    ap.add_argument("--zinc-sample-n", type=int, default=15000)
    ap.add_argument("--gen-sample-n", type=int, default=3000)
    ap.add_argument("--default-glob", default="runs/replicated_eval_pcmc_fixed_1/rep_*/trial_1.csv")
    ap.add_argument("--optimized-glob", default="runs/replicated_eval_pcmc_fixed_optimized_1/rep_*/trial_1.csv")
    ap.add_argument("--out", default="pcmc_surften_scatter.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv, usecols=["pCMC", "SurfTen"])
    holdout = pd.read_csv(args.holdout_csv, usecols=["pCMC", "SurfTen", "SMILES_canonical"])
    holdout["canon"] = holdout["SMILES_canonical"].apply(canon)

    print("loading ZINC scored dataset (sampling)...", flush=True)
    zinc_full = pd.read_csv(args.zinc_scored, usecols=["valid", "pCMC_mean", "surface_tension_avg_mean"])
    zinc_full = zinc_full[zinc_full["valid"]]
    zinc = zinc_full.sample(n=min(args.zinc_sample_n, len(zinc_full)), random_state=args.seed)
    zinc = zinc.rename(columns={"pCMC_mean": "pCMC", "surface_tension_avg_mean": "SurfTen"})

    zinc_best = pd.read_csv(args.zinc_best_csv, usecols=["pCMC_mean", "surface_tension_avg_mean"])
    zinc_best = zinc_best.rename(columns={"pCMC_mean": "pCMC", "surface_tension_avg_mean": "SurfTen"})

    default_gen = load_generated(args.default_glob)
    optimized_gen = load_generated(args.optimized_glob)

    print(f"train={len(train)}  holdout={len(holdout)}  zinc_sample={len(zinc)}  "
          f"zinc_best={len(zinc_best)}  default_gen={len(default_gen)}  "
          f"optimized_gen={len(optimized_gen)}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    plot_panel(ax1, "Default hyperparameters", default_gen, args.gen_sample_n,
               train, holdout, zinc, zinc_best, args.seed)
    plot_panel(ax2, "Optimized hyperparameters", optimized_gen, args.gen_sample_n,
               train, holdout, zinc, zinc_best, args.seed)

    fig.suptitle("pCMC vs. surface tension at reference concentration: SurfPro, ZINC, and generated molecules\n"
                 "(corrected pCMC direction, 2026-07-22)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
