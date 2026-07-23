#!/usr/bin/env python3
"""
Build a quality-stratified SurfPro-MD train/holdout split (26 molecules per
quintile tier => 130 holdout, 1421 trainval), ranked by a composite of real
measured pCMC and SurfTen.

pCMC is -log10(CMC in mol/L) (pH-style; confirmed empirically 2026-07-21 --
e.g. dodecyl sulfate/SDS has pCMC=2.03 in SurfPro-MD.csv, implying
CMC=9.3 mM, matching SDS's known experimental CMC). HIGHER pCMC means LOWER
CMC, i.e. a MORE efficient surfactant -- so pCMC should be MAXIMIZED, not
minimized. An earlier version of this composite (and of config.json's RL
objective) treated pCMC as "lower is better" like SurfTen, which is backwards.
This script (and the fixed config.json/build_zinc_*.py) correct that.

Usage:
    python workflow/build_surfpro_stratified_holdout.py \
        --input data/surfpro_expanded.csv \
        --holdout-out data/surfpro_real_holdout_test_split.csv \
        --trainval-out data/surfpro_expanded_trainval_only.csv \
        --n-per-tier 26 --seed 42
"""
import argparse

import numpy as np
import pandas as pd


def norm_high_is_good(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def norm_low_is_good(x, lo, hi):
    return 1 - np.clip((x - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/surfpro_expanded.csv")
    ap.add_argument("--holdout-out", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--trainval-out", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--n-per-tier", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"loaded {len(df)} molecules from {args.input}", flush=True)

    p5_pcmc, p95_pcmc = df["pCMC"].quantile([0.05, 0.95])
    p5_st, p95_st = df["SurfTen"].quantile([0.05, 0.95])
    print(f"pCMC p5/p95: {p5_pcmc:.4f} / {p95_pcmc:.4f}", flush=True)
    print(f"SurfTen p5/p95: {p5_st:.4f} / {p95_st:.4f}", flush=True)

    s_pcmc = norm_high_is_good(df["pCMC"], p5_pcmc, p95_pcmc)   # HIGH pCMC = good (low CMC)
    s_surften = norm_low_is_good(df["SurfTen"], p5_st, p95_st)  # LOW SurfTen = good
    df["true_composite"] = (s_pcmc * s_surften) ** 0.5

    df["quality_tier"] = pd.qcut(
        df["true_composite"].rank(ascending=False, method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    print(df.groupby("quality_tier", observed=True).size(), flush=True)

    rng = np.random.RandomState(args.seed)
    holdout_parts = []
    for tier in [1, 2, 3, 4, 5]:
        tier_df = df[df["quality_tier"] == tier]
        idx = rng.choice(tier_df.index, size=args.n_per_tier, replace=False)
        holdout_parts.append(df.loc[idx])
    holdout = pd.concat(holdout_parts).sort_index()
    trainval = df.drop(holdout.index)

    holdout.to_csv(args.holdout_out, index=False)
    trainval.to_csv(args.trainval_out, index=False)
    print(f"holdout: {len(holdout)} -> {args.holdout_out}", flush=True)
    print(f"trainval: {len(trainval)} -> {args.trainval_out}", flush=True)
    print(holdout["quality_tier"].value_counts().sort_index(), flush=True)


if __name__ == "__main__":
    main()
