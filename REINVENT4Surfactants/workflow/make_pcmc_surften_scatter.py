#!/usr/bin/env python3
"""
Publication-ready SurfTen-vs-pCMC scatter figure for the production sweep,
restricted to the ZINC-similarity-off, uncertainty-mode in {none, lm}
combinations (2026-07-31: SM and SM+LM dropped as not effective; ZINC-
similarity excluded as a plotted dimension). One figure, 2 columns
(uncertainty mode none/LM) x 2 rows -- top row a random 2000-molecule
subsample per Pareto arm (pooled across all replicates), bottom row the
TOP 2000 by Score per Pareto arm (2026-08-04: a random subsample can miss
most of the best-scoring molecules, which are exactly the interesting ones
for a property-space plot). No plot/subplot titles. Each panel overlays
three populations: the SurfPro-MD training set, the SurfPro holdout, and the
generated molecules (predicted, in the surrogate models' native units,
inverting REINVENT's [0,1] score normalization using config.json's
calibration bounds).

Usage:
    python workflow/make_pcmc_surften_scatter.py \
        --combos-dir runs/production --out-dir figures
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UNC_ORDER = ["none", "lm"]
UNC_SUFFIX = {"none": "", "lm": " + LM"}
PARETO_ORDER = ["none", "boost", "gradient"]
PARETO_LABELS = {"none": "no Pareto", "boost": "ParetoBoost", "gradient": "ParetoGradient"}

# Viridis, sampled at 3 well-separated points (requested 2026-08-04, replacing
# the earlier CVD-validated categorical triplet).
COLORS = {"none": "#471365", "boost": "#21918c", "gradient": "#bddf26"}
HOLDOUT_COLOR = "#d62728"
TRAIN_COLOR = "#0b0b0b"

GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

N_POINTS = 2000
SEED = 0

XLABEL = "SurfTen (arb. units)"
YLABEL = "pCMC (−log₁₀ mol/L)"


def invert_score(score, min_value, max_value, minimize):
    score = np.asarray(score, dtype=float)
    if minimize:
        return min_value + (1 - score) * (max_value - min_value)
    return min_value + score * (max_value - min_value)


def load_combo_pool(combo_dir, bounds):
    """Pooled (pcmc, surften, Score) across every accepted replicate for this
    combo, artifact-filtered, un-subsampled -- both the random and top-N
    selections are derived from this same pool."""
    dfs = []
    for rep_csv in sorted(glob.glob(f"{combo_dir}/production/rep_*/trial_1.csv")):
        # Skip partial leftovers from replicates the orchestrator didn't
        # accept (no eval.json) -- see make_production_stepwise_figures.py.
        if not os.path.exists(os.path.join(os.path.dirname(rep_csv), "eval.json")):
            continue
        df = pd.read_csv(rep_csv, usecols=["pCMC (raw)", "SurfTen (raw)", "Score"])
        dfs.append(df)
    if not dfs:
        return None
    all_df = pd.concat(dfs, ignore_index=True).dropna()
    # Drop scoring-failure artifacts: extreme/out-of-distribution structures where
    # the surrogate pipeline fails and REINVENT floors every component (and Score)
    # to exactly 0, rather than a genuinely poor prediction.
    all_df = all_df[~((all_df["pCMC (raw)"] == 0) & (all_df["SurfTen (raw)"] == 0) & (all_df["Score"] == 0))]
    pcmc = invert_score(all_df["pCMC (raw)"], *bounds["pCMC"])
    surften = invert_score(all_df["SurfTen (raw)"], *bounds["SurfTen"])
    return pd.DataFrame({"pCMC": pcmc, "SurfTen": surften, "Score": all_df["Score"].to_numpy()})


def select_random(pool, n, rng):
    if len(pool) > n:
        idx = rng.choice(len(pool), n, replace=False)
        return pool.iloc[idx]
    return pool


def select_top(pool, n):
    return pool.sort_values("Score", ascending=False).head(n)


def style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_INK)
    ax.tick_params(labelsize=9, colors=MUTED_INK)
    ax.xaxis.label.set_color(PRIMARY_INK)
    ax.yaxis.label.set_color(PRIMARY_INK)


def plot_panel(ax, pools, train_df, holdout_df, unc, select_fn):
    # Layering (bottom -> top): training set, generated molecules, holdout.
    ax.scatter(train_df["SurfTen"], train_df["pCMC"], s=14, alpha=0.65, linewidths=0,
               marker="D", color=TRAIN_COLOR, label="Training set", zorder=2)
    for pareto in PARETO_ORDER:
        pool = pools[(unc, pareto)]
        if pool is None:
            continue
        sel = select_fn(pool)
        ax.scatter(sel["SurfTen"], sel["pCMC"], s=6, alpha=0.55, linewidths=0,
                   color=COLORS[pareto], label=f"{PARETO_LABELS[pareto]}{UNC_SUFFIX[unc]}", zorder=3)
    ax.scatter(holdout_df["SurfTen"], holdout_df["pCMC"], s=32, marker="*",
               color=HOLDOUT_COLOR, label="Holdout test set", zorder=5, linewidths=0)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="best", markerscale=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos-dir", default="runs/production")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config) as f:
        cfg = json.load(f)
    sf = cfg["SCORING_FUNCTIONS"]
    bounds = {
        "pCMC": (sf["pCMC"]["min_value"], sf["pCMC"]["max_value"], sf["pCMC"]["minimize"]),
        "SurfTen": (sf["SurfTen"]["min_value"], sf["SurfTen"]["max_value"], sf["SurfTen"]["minimize"]),
    }

    train_df = pd.read_csv(args.train_csv, usecols=["pCMC", "SurfTen"]).dropna()
    holdout_df = pd.read_csv(args.surfpro_holdout, usecols=["pCMC", "SurfTen"]).dropna()

    pools = {}
    for unc in UNC_ORDER:
        for pareto in PARETO_ORDER:
            combo_dir = f"{args.combos_dir}/zinc_off-unc_{unc}-pareto_{pareto}"
            print(f"loading {os.path.basename(combo_dir)}...", flush=True)
            pools[(unc, pareto)] = load_combo_pool(combo_dir, bounds)

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, unc in enumerate(UNC_ORDER):
        plot_panel(axes[0][col], pools, train_df, holdout_df, unc,
                   lambda pool: select_random(pool, N_POINTS, rng))
        plot_panel(axes[1][col], pools, train_df, holdout_df, unc,
                   lambda pool: select_top(pool, N_POINTS))

    fig.tight_layout()
    out_path = f"{args.out_dir}/scatter_pcmc_surften.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
