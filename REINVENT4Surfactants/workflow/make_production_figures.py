#!/usr/bin/env python3
"""
Build comparison figures for the 24-combination production sweep from
runs/production/comparison_table.csv (produced by gather_production_results.py).

Usage:
    python workflow/make_production_figures.py \
        --table runs/production/comparison_table.csv --out-dir figures
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UNC_ORDER = ["none", "sm", "lm", "sm_lm"]
UNC_LABELS = {"none": "None", "sm": "SM", "lm": "LM", "sm_lm": "SM+LM"}
PARETO_ORDER = ["none", "boost", "gradient"]
PARETO_LABELS = {"none": "None", "boost": "ParetoBoost", "gradient": "ParetoGradient"}
ZINC_ORDER = ["on", "off"]
ZINC_LABELS = {"on": "ZINC-similarity on", "off": "ZINC-similarity off"}
COLORS = {"none": "#9E9E9E", "boost": "#7B539E", "gradient": "#5A9448"}


def grouped_bar_panel(ax, df, metric, title, ylabel, ylim=None):
    x = np.arange(len(UNC_ORDER))
    width = 0.25
    for i, pareto in enumerate(PARETO_ORDER):
        means, stds = [], []
        for unc in UNC_ORDER:
            row = df[(df.unc_mode == unc) & (df.pareto_mode == pareto)]
            means.append(row[f"{metric}_mean"].values[0] if len(row) else np.nan)
            stds.append(row[f"{metric}_std"].values[0] if len(row) else np.nan)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=2,
               label=PARETO_LABELS[pareto], color=COLORS[pareto])
    ax.set_xticks(x)
    ax.set_xticklabels([UNC_LABELS[u] for u in UNC_ORDER])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)


def make_figure(df, metrics, out_path, figsize):
    fig, axes = plt.subplots(len(ZINC_ORDER), len(metrics), figsize=figsize, squeeze=False)
    for row, zinc in enumerate(ZINC_ORDER):
        sub = df[df.zinc == zinc]
        for col, (metric, title, ylabel, ylim) in enumerate(metrics):
            ax = axes[row][col]
            grouped_bar_panel(ax, sub, metric, title if row == 0 else "", ylabel, ylim)
            if col == 0:
                ax.annotate(ZINC_LABELS[zinc], xy=(-0.45, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", ha="right", va="center",
                            rotation=90)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.03, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="runs/production/comparison_table.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.table)

    make_figure(
        df,
        [
            ("renormalized_score", "Renormalized score\n(pCMC + SurfTen only)", "score", (0, 0.65)),
            ("surfpro_top100", "SurfPro top-100\nrediscovery rate", "rate", (0, 0.02)),
            ("zinc_top100", "ZINC top-100\nrediscovery rate", "rate", (0, 0.30)),
            ("nn_tanimoto_to_train", "NN Tanimoto similarity\nto training set", "similarity", (0, 1.0)),
        ],
        f"{args.out_dir}/production_sweep_scores.png",
        figsize=(13, 6),
    )

    make_figure(
        df,
        [
            ("novelty", "Novelty", "fraction", (0.8, 1.0)),
            ("internal_diversity", "Internal diversity", "1 - mean Tanimoto", (0.55, 0.8)),
            ("validity", "Validity", "fraction", (0.95, 1.0)),
        ],
        f"{args.out_dir}/production_sweep_diversity.png",
        figsize=(10, 6),
    )


if __name__ == "__main__":
    main()
