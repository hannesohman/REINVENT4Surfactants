#!/usr/bin/env python3
"""
Scatter plots of (predicted) SurfTen vs. pCMC for every generated molecule in
the 24-combination production sweep, in the surrogate models' native units
(inverting REINVENT's [0,1] score normalization using the calibration bounds
in config.json), with the real SurfPro top-100 holdout overlaid as reference.

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

UNC_ORDER = ["none", "sm", "lm", "sm_lm"]
UNC_LABELS = {"none": "None", "sm": "SM", "lm": "LM", "sm_lm": "SM+LM"}
PARETO_ORDER = ["none", "boost", "gradient"]
PARETO_LABELS = {"none": "None", "boost": "ParetoBoost", "gradient": "ParetoGradient"}
ZINC_ORDER = ["on", "off"]
ZINC_LABELS = {"on": "ZINC-similarity on", "off": "ZINC-similarity off"}

# Validated categorical triplet (dataviz skill palette.md, slots 1-3: blue/orange/
# aqua) -- the only 3-color subset of that palette confirmed to clear the CVD /
# normal-vision floors under all-pairs comparison, which a scatter is.
COLORS = {"none": "#2a78d6", "boost": "#eb6834", "gradient": "#1baf7a"}
HOLDOUT_COLOR = "#0b0b0b"

N_SUBSAMPLE = 1500
SEED = 0


def invert_score(score, min_value, max_value, minimize):
    score = np.asarray(score, dtype=float)
    if minimize:
        return min_value + (1 - score) * (max_value - min_value)
    return min_value + score * (max_value - min_value)


def load_combo_points(combo_dir, bounds, rng):
    dfs = []
    for rep_csv in sorted(glob.glob(f"{combo_dir}/production/rep_*/trial_1.csv")):
        df = pd.read_csv(rep_csv, usecols=["pCMC (raw)", "SurfTen (raw)", "Score"])
        dfs.append(df)
    if not dfs:
        return None, None
    all_df = pd.concat(dfs, ignore_index=True).dropna()
    # Drop scoring-failure artifacts: extreme/out-of-distribution structures where
    # the surrogate pipeline fails and REINVENT floors every component (and Score)
    # to exactly 0, rather than a genuinely poor prediction.
    all_df = all_df[~((all_df["pCMC (raw)"] == 0) & (all_df["SurfTen (raw)"] == 0) & (all_df["Score"] == 0))]
    pcmc = invert_score(all_df["pCMC (raw)"], *bounds["pCMC"])
    surften = invert_score(all_df["SurfTen (raw)"], *bounds["SurfTen"])
    n = len(pcmc)
    if n > N_SUBSAMPLE:
        idx = rng.choice(n, N_SUBSAMPLE, replace=False)
        pcmc, surften = pcmc[idx], surften[idx]
    return pcmc, surften


def make_figure(combos_dir, bounds, holdout, zinc, out_path):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(UNC_ORDER), figsize=(15, 4), sharex=True, sharey=True)

    for col, unc in enumerate(UNC_ORDER):
        ax = axes[col]
        ax.scatter(holdout["pCMC"], holdout["SurfTen"], s=22, marker="*",
                   color=HOLDOUT_COLOR, label="SurfPro top-100 (ground truth)",
                   zorder=5, linewidths=0)
        for pareto in PARETO_ORDER:
            combo_dir = f"{combos_dir}/zinc_{zinc}-unc_{unc}-pareto_{pareto}"
            pcmc, surften = load_combo_points(combo_dir, bounds, rng)
            if pcmc is None:
                continue
            ax.scatter(pcmc, surften, s=6, alpha=0.35, linewidths=0,
                       color=COLORS[pareto], label=PARETO_LABELS[pareto])
        ax.set_title(UNC_LABELS[unc], fontsize=11)
        ax.set_xlabel("pCMC (predicted)", fontsize=9)
        ax.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("SurfTen (predicted)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), markerscale=2, frameon=False)
    fig.suptitle(f"{ZINC_LABELS[zinc]} -- lower SurfTen & higher pCMC is better "
                 "(bottom-right)", fontsize=11, y=1.03)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos-dir", default="runs/production")
    ap.add_argument("--config", default="config.json")
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

    holdout = pd.read_csv(args.surfpro_holdout, usecols=["pCMC", "SurfTen"]).dropna()

    for zinc in ZINC_ORDER:
        make_figure(args.combos_dir, bounds, holdout, zinc,
                    f"{args.out_dir}/pcmc_surften_scatter_zinc_{zinc}.png")


if __name__ == "__main__":
    main()
