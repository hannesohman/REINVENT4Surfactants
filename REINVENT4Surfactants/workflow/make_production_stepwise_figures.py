#!/usr/bin/env python3
"""
Publication-ready property-vs-RL-step figures for the production sweep,
restricted to the ZINC-similarity-off, uncertainty-mode in {none, lm}
combinations (2026-07-31: SM and SM+LM dropped as not effective; ZINC-
similarity excluded as a plotted dimension throughout). One standalone figure
per metric (no plot/subplot titles -- panel identity carries entirely in the
legend), with gridlines and shaded +/-1 std error bands across the 5
replicates. Metrics: renormalized score, validity, novelty, internal
diversity, and nearest-neighbor Tanimoto DISTANCE to the SurfPro top-100
HOLDOUT set (not the training set).

Usage:
    python workflow/make_production_stepwise_figures.py \
        --combos-dir runs/production --out-dir figures
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
# dataviz skill categorical palette, slots 1-3 (blue/orange/aqua) -- the only
# 3-color subset validated all-pairs (CVD + normal-vision) for scatter/small-
# multiples use; reused here for consistency across all production figures.
COLORS = {"none": "#2a78d6", "boost": "#eb6834", "gradient": "#1baf7a"}

GRIDLINE = "#e1e0d9"
AXIS_INK = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

METRICS = [
    ("renormalized_score", "Renormalized score", "stepwise_renormalized_score.png"),
    ("validity", "Validity (fraction)", "stepwise_validity.png"),
    ("novelty", "Novelty (fraction)", "stepwise_novelty.png"),
    ("internal_diversity", "Internal diversity (1 − mean Tanimoto similarity)", "stepwise_internal_diversity.png"),
    ("nn_tanimoto_dist_to_holdout", "NN Tanimoto distance to holdout", "stepwise_tanimoto_dist_holdout.png"),
]


def step_metrics(step_df: pd.DataFrame, trainval_set: set, holdout_fps: list) -> dict:
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


def load_combo_trajectory(combo_dir: str, trainval_set: set, holdout_fps: list) -> pd.DataFrame:
    rows = []
    for rep_csv in sorted(glob.glob(f"{combo_dir}/production/rep_*/trial_1.csv")):
        df = pd.read_csv(rep_csv, usecols=["SMILES", "Score", "pCMC (raw)", "SurfTen (raw)", "step"])
        for step, step_df in df.groupby("step"):
            m = step_metrics(step_df, trainval_set, holdout_fps)
            m["step"] = step
            rows.append(m)
    return pd.DataFrame(rows)


def make_legend(ax):
    handles = []
    for pareto in PARETO_ORDER:
        handles.append(plt.Line2D([0], [0], color=COLORS[pareto], linewidth=2, label=PARETO_LABELS[pareto]))
    for unc in UNC_ORDER:
        handles.append(plt.Line2D([0], [0], color=PRIMARY_INK, linewidth=2,
                                   linestyle=UNC_LINESTYLE[unc], label=UNC_LABELS[unc]))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")


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
            traj = load_combo_trajectory(combo_dir, trainval_set, holdout_fps)
            trajectories[(unc, pareto)] = traj.groupby("step").agg(["mean", "std"])

    for metric, ylabel, filename in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for unc in UNC_ORDER:
            for pareto in PARETO_ORDER:
                agg = trajectories[(unc, pareto)]
                steps = agg.index.to_numpy()
                mean = agg[(metric, "mean")].to_numpy()
                std = agg[(metric, "std")].to_numpy()
                ax.plot(steps, mean, color=COLORS[pareto], linestyle=UNC_LINESTYLE[unc], linewidth=2)
                lo = np.clip(mean - std, 0, 1)
                hi = np.clip(mean + std, 0, 1)
                ax.fill_between(steps, lo, hi, color=COLORS[pareto], alpha=0.15, linewidth=0)
        ax.set_xlabel("RL step")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        make_legend(ax)
        fig.tight_layout()
        out_path = f"{args.out_dir}/{filename}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
