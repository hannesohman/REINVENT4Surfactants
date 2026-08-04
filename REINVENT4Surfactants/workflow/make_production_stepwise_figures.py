#!/usr/bin/env python3
"""
Publication-ready property-vs-molecules-generated figure for the production
sweep, restricted to the ZINC-similarity-off, uncertainty-mode in {none, lm}
combinations (2026-07-31: SM and SM+LM dropped as not effective; ZINC-
similarity excluded as a plotted dimension throughout). One figure, one
subplot per metric (no plot/subplot titles -- panel identity carries in the
y-axis label; a single shared legend), with gridlines and shaded +/-1 std
error bands across replicates. Metrics: renormalized score, validity,
novelty, internal diversity, and nearest-neighbor Tanimoto DISTANCE to the
SurfPro top-100 HOLDOUT set (not the training set).

Metrics are computed over fixed-size bins of --bin-size (default 100)
consecutively-generated molecules, not raw RL steps (2026-08-04): since step
count is derived per combination from a fixed oracle budget divided by that
combination's own HPO-chosen batch size, raw per-step batches range from 10
to 500 molecules across combinations, making small-batch trajectories very
noisy (each point a statistic over only ~10 molecules) and step counts
incomparable across lines. Binning by molecule-generation order instead
(splitting large steps, merging consecutive small ones) gives every
combination the same, comparably-sized sample per plotted point.

Usage:
    python workflow/make_production_stepwise_figures.py \
        --combos-dir runs/production --out-dir figures --bin-size 100
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_run import canon, morgan_fp, internal_diversity, nn_tanimoto_similarity  # noqa: E402

RDLogger.DisableLog("rdApp.*")

UNC_ORDER = ["none", "lm"]
UNC_LABELS = {"none": "Uncertainty: none", "lm": "Uncertainty: LM"}
UNC_LINESTYLE = {"none": "-", "lm": "--"}
PARETO_ORDER = ["none", "boost", "gradient"]
PARETO_LABELS = {"none": "Pareto: none", "boost": "Pareto: ParetoBoost", "gradient": "Pareto: ParetoGradient"}
# Viridis, sampled at 3 well-separated points (requested 2026-08-04, replacing
# the earlier CVD-validated categorical triplet).
COLORS = {"none": "#471365", "boost": "#21918c", "gradient": "#bddf26"}

GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

METRICS = [
    ("renormalized_score", "Renormalized score"),
    ("novelty", "Novelty (fraction)"),
    ("internal_diversity", "Internal diversity (1 − mean Tanimoto similarity)"),
    ("nn_tanimoto_dist_to_holdout", "NN Tanimoto distance to holdout"),
    ("validity", "Validity (fraction)"),
]
GRID_SHAPE = (2, 3)  # 5 metrics + 1 shared-legend panel


def bin_metrics(step_df: pd.DataFrame, trainval_set: set, holdout_fps: list) -> dict:
    n_total = len(step_df)
    canons = step_df["SMILES"].apply(canon)
    n_valid = int(canons.notna().sum())
    validity = n_valid / n_total if n_total else float("nan")

    valid_canons = canons.dropna().tolist()
    novel = sum(1 for s in valid_canons if s not in trainval_set)
    novelty = novel / n_valid if n_valid else float("nan")

    valid_mols = [m for m in (Chem.MolFromSmiles(s) for s in valid_canons) if m is not None]
    intdiv = internal_diversity(valid_mols, sample_n=len(valid_mols))
    nn_dist = 1 - nn_tanimoto_similarity(valid_mols, holdout_fps)

    p = step_df["pCMC (raw)"].to_numpy(dtype=float)
    s = step_df["SurfTen (raw)"].to_numpy(dtype=float)
    sc = step_df["Score"].to_numpy(dtype=float)
    artifact = (p == 0) & (s == 0) & (sc == 0)
    ok = np.isfinite(p) & np.isfinite(s) & ~artifact
    renorm = float(np.sqrt(np.clip(p[ok], 0, None) * np.clip(s[ok], 0, None)).mean()) if ok.any() else float("nan")

    return {
        "renormalized_score": renorm,
        "validity": validity,
        "novelty": novelty,
        "internal_diversity": intdiv,
        "nn_tanimoto_dist_to_holdout": nn_dist,
    }


def load_combo_trajectory(combo_dir: str, trainval_set: set, holdout_fps: list, bin_size: int) -> pd.DataFrame:
    rows = []
    for rep_csv in sorted(glob.glob(f"{combo_dir}/production/rep_*/trial_1.csv")):
        # Only replicates run_production_combo.py itself accepted as
        # successful (eval.json written) -- a rep_*/trial_1.csv can be a
        # partial leftover from a replicate that failed/was abandoned
        # mid-generation, which the orchestrator skips but doesn't delete
        # (found 2026-08-04: one such partial file was skewing a trajectory).
        if not os.path.exists(os.path.join(os.path.dirname(rep_csv), "eval.json")):
            print(f"  skipping {rep_csv} (no eval.json -- not a completed replicate)", flush=True)
            continue
        df = pd.read_csv(rep_csv, usecols=["SMILES", "Score", "pCMC (raw)", "SurfTen (raw)", "step"])
        # Bin by molecule-generation order (row position), not by raw RL
        # step -- rows are already in generation order (sequential steps,
        # sequential within-step order), so integer-dividing the row's
        # position by a fixed bin_size transparently merges consecutive
        # steps for small batch sizes and splits a step for large ones.
        df = df.reset_index(drop=True)
        bin_idx = df.index // bin_size
        for b, bin_df in df.groupby(bin_idx):
            m = bin_metrics(bin_df, trainval_set, holdout_fps)
            m["bin"] = b
            rows.append(m)
    return pd.DataFrame(rows)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos-dir", default="runs/production")
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--train-smiles-col", default="SMILES_canonical")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--bin-size", type=int, default=100,
                     help="molecules per plotted point, regardless of each combo's own batch size")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    trainval_set = set(train_df[args.train_smiles_col].apply(canon).dropna())

    holdout_df = pd.read_csv(args.surfpro_holdout)
    holdout_mols = [m for m in (Chem.MolFromSmiles(s) for s in holdout_df["SMILES_canonical"]) if m is not None]
    holdout_fps = [morgan_fp(m) for m in holdout_mols]
    print(f"trainval: {len(trainval_set)} molecules, holdout: {len(holdout_fps)} molecules", flush=True)

    trajectories = {}
    for unc in UNC_ORDER:
        for pareto in PARETO_ORDER:
            name = f"zinc_off-unc_{unc}-pareto_{pareto}"
            combo_dir = f"{args.combos_dir}/{name}"
            print(f"processing {name}...", flush=True)
            traj = load_combo_trajectory(combo_dir, trainval_set, holdout_fps, args.bin_size)
            trajectories[(unc, pareto)] = traj.groupby("bin").agg(["mean", "std"])

    nrows, ncols = GRID_SHAPE
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for ax, (metric, ylabel) in zip(axes_flat, METRICS):
        for unc in UNC_ORDER:
            for pareto in PARETO_ORDER:
                agg = trajectories[(unc, pareto)]
                # Bin index -> cumulative molecules generated (bins are a
                # fixed args.bin_size molecules each, comparable across every
                # combination regardless of its own HPO-chosen batch size).
                molecules = (agg.index.to_numpy() + 1) * args.bin_size
                mean = agg[(metric, "mean")].to_numpy()
                std = agg[(metric, "std")].to_numpy()
                ax.plot(molecules, mean, color=COLORS[pareto], linestyle=UNC_LINESTYLE[unc], linewidth=2)
                lo = np.clip(mean - std, 0, 1)
                hi = np.clip(mean + std, 0, 1)
                ax.fill_between(molecules, lo, hi, color=COLORS[pareto], alpha=0.15, linewidth=0)
        ax.set_xlabel("Molecules generated")
        ax.set_ylabel(ylabel)
        style_axes(ax)

    # Remaining (unused) grid slots host the single shared legend instead of
    # a per-panel one -- color/linestyle mean the same thing in every panel.
    handles = []
    for pareto in PARETO_ORDER:
        handles.append(plt.Line2D([0], [0], color=COLORS[pareto], linewidth=2, label=PARETO_LABELS[pareto]))
    for unc in UNC_ORDER:
        handles.append(plt.Line2D([0], [0], color=PRIMARY_INK, linewidth=2,
                                   linestyle=UNC_LINESTYLE[unc], label=UNC_LABELS[unc]))
    for ax in axes_flat[len(METRICS):]:
        ax.axis("off")
    axes_flat[len(METRICS)].legend(handles=handles, frameon=False, fontsize=11, loc="center")

    fig.tight_layout()
    out_path = f"{args.out_dir}/stepwise_all.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
