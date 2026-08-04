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
import gzip
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

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
GRID_SHAPE = (2, 3)  # 5 metrics + 1 rediscovery-rate bar panel

REDISCOVERY_SOURCES = ["surfpro", "zinc", "chembl"]
REDISCOVERY_LABELS = {"surfpro": "SurfPro", "zinc": "ZINC", "chembl": "ChEMBL"}
CHEMBL_TOP_PCT = 5.0  # same percentile convention as the HPO objective


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


def load_chembl_reference(path):
    with gzip.open(path, "rt") as f:
        ref = json.load(f)
    return set(ref["inchikeys_full"]), set(ref["inchikeys_skeleton"])


def to_inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def chembl_replicate_rate(rep_csv, full_set, skeleton_set, top_pct=CHEMBL_TOP_PCT):
    """Fraction of the top top_pct% (by Score) unique valid molecules in this
    replicate that already exist in ChEMBL (exact InChIKey or skeleton-only
    match) -- same percentile convention as the HPO objective."""
    df = pd.read_csv(rep_csv, usecols=["SMILES", "Score"])
    valid = df["SMILES"].apply(lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    df = df[valid]
    unique_df = df.drop_duplicates(subset=["SMILES"])
    n_top = max(1, round(len(unique_df) * top_pct / 100))
    top_df = unique_df.sort_values("Score", ascending=False).head(n_top)
    keys = top_df["SMILES"].apply(to_inchikey).dropna()
    if keys.empty:
        return float("nan")
    skeletons = keys.str[:14]
    hits = keys.isin(full_set) | skeletons.isin(skeleton_set)
    return float(hits.mean())


def load_rediscovery_rates(combo_dir, full_set, skeleton_set):
    """Per-replicate SurfPro/ZINC/ChEMBL rediscovery rates for one
    combination -- SurfPro/ZINC come straight from each replicate's own
    eval.json (already computed by run_production_combo.py); ChEMBL is
    computed fresh here (never run against real production output before)."""
    rates = {src: [] for src in REDISCOVERY_SOURCES}
    for rep_csv in sorted(glob.glob(f"{combo_dir}/production/rep_*/trial_1.csv")):
        eval_path = os.path.join(os.path.dirname(rep_csv), "eval.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            result = json.load(f)
        rates["surfpro"].append(result["surfpro_top100"]["rate"])
        rates["zinc"].append(result["zinc_top100"]["rate"])
        rates["chembl"].append(chembl_replicate_rate(rep_csv, full_set, skeleton_set))
    return {src: (float(np.mean(v)), float(np.std(v))) for src, v in rates.items()}


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
    ap.add_argument("--chembl-reference", default="data/chembl_reference.json.gz")
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

    print(f"loading ChEMBL reference from {args.chembl_reference} ...", flush=True)
    chembl_full, chembl_skeleton = load_chembl_reference(args.chembl_reference)

    trajectories = {}
    rediscovery = {}
    for unc in UNC_ORDER:
        for pareto in PARETO_ORDER:
            name = f"zinc_off-unc_{unc}-pareto_{pareto}"
            combo_dir = f"{args.combos_dir}/{name}"
            print(f"processing {name}...", flush=True)
            traj = load_combo_trajectory(combo_dir, trainval_set, holdout_fps, args.bin_size)
            trajectories[(unc, pareto)] = traj.groupby("bin").agg(["mean", "std"])
            rediscovery[(unc, pareto)] = load_rediscovery_rates(combo_dir, chembl_full, chembl_skeleton)

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

    # Last grid slot: mean rediscovery rate per source (SurfPro/ZINC/ChEMBL),
    # grouped bars per combination, +/-1 std error bars across replicates.
    # Color = Pareto mode (matches the line panels); hatch = uncertainty mode
    # (solid fill = none, diagonal hatch = LM), since color alone is already
    # used for Pareto mode in this figure.
    rax = axes_flat[len(METRICS)]
    combos = [(unc, pareto) for unc in UNC_ORDER for pareto in PARETO_ORDER]
    n_bars = len(combos)
    bar_width = 0.8 / n_bars
    group_x = np.arange(len(REDISCOVERY_SOURCES))
    for i, (unc, pareto) in enumerate(combos):
        means = [rediscovery[(unc, pareto)][src][0] for src in REDISCOVERY_SOURCES]
        stds = [rediscovery[(unc, pareto)][src][1] for src in REDISCOVERY_SOURCES]
        x = group_x + (i - (n_bars - 1) / 2) * bar_width
        rax.bar(x, means, bar_width, yerr=stds, capsize=2, color=COLORS[pareto],
                hatch="////" if unc == "lm" else None, edgecolor="white", linewidth=0.5)
    rax.set_xticks(group_x)
    rax.set_xticklabels([REDISCOVERY_LABELS[s] for s in REDISCOVERY_SOURCES])
    rax.set_ylabel("Rediscovery rate")
    style_axes(rax)
    rax.grid(axis="x", visible=False)

    for ax in axes_flat[len(METRICS) + 1:]:
        ax.axis("off")

    # Single shared legend for the whole figure (line style/hatch both mean
    # "uncertainty mode"; color means "Pareto mode" in every panel).
    handles = []
    for pareto in PARETO_ORDER:
        handles.append(plt.Line2D([0], [0], color=COLORS[pareto], linewidth=2, label=PARETO_LABELS[pareto]))
    for unc in UNC_ORDER:
        handles.append(plt.Line2D([0], [0], color=PRIMARY_INK, linewidth=2,
                                   linestyle=UNC_LINESTYLE[unc], label=f"{UNC_LABELS[unc]} (lines)"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PRIMARY_INK, label="Uncertainty: none (bars)"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=PRIMARY_INK, hatch="////", label="Uncertainty: LM (bars)"))
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    out_path = f"{args.out_dir}/stepwise_all.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
