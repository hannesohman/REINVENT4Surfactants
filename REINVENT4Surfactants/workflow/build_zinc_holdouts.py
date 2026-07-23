#!/usr/bin/env python3
"""
Build the ZINC-derived holdout sets used to sanity-check generative runs, from
the output of score_zinc_surrogates.py (data/ZINC/zinc_scored_9props.csv.gz).

Two outputs:
  1. data/zinc_holdout_low_pCMC_low_SurfTen.csv -- a small (~400 molecule,
     pre-filter) "best of the best" list, ranked by a composite score.
  2. data/zinc_holdout_top{5,10,15,20}pct.csv.gz -- staggered, increasingly
     lenient tiers (nested: top5% subset of top10% subset of ...), for
     checking what fraction of generated molecules land in each tier.

The composite score is an equal-weighted geometric mean of four normalized
terms -- pCMC (maximized: higher pCMC = lower CMC = more efficient
surfactant), SurfTen (minimized), and their ensemble-std "uncertainty" terms
(minimized -- low disagreement = reliable) -- matching exactly the RL
objective in config.json (WEIGHT_COMBOS: pCMC/SurfTen/pCMC_Uncertainty/
SurfTen_Uncertainty, 0.25 each; pCMC direction fixed 2026-07-21, see README).
Property terms are normalized against the *eligible ZINC population's own*
5th/95th percentile; uncertainty terms are normalized against config.json's
pCMC_Uncertainty/SurfTen_Uncertainty min_value/max_value (the SurfPro-MD
training set's 5th/95th percentile ensemble std) for methodological parity
with the RL objective.

Before ranking, molecules containing any element absent from the SurfPro-MD
training set (C, Cl, F, N, O, P, S, Si) are excluded -- real, synthesizable
molecules only, but not restricted to any surfactant-like substructure.

Usage:
    python workflow/build_zinc_holdouts.py \
        --scored data/ZINC/zinc_scored_9props.csv.gz \
        --train-csv /path/to/SurfPro-MD/SurfPro-MD.csv \
        --out-dir data \
        --top-n 400
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PCMC_UNC_MIN, PCMC_UNC_MAX = 0.0476, 0.6120
SURFTEN_UNC_MIN, SURFTEN_UNC_MAX = 2.806, 18.979


def get_training_elements(train_csv: str) -> set[str]:
    train_df = pd.read_csv(train_csv, usecols=["SMILES_canonical"])
    elements = set()
    for smi in train_df["SMILES_canonical"].dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        elements.update(atom.GetSymbol() for atom in mol.GetAtoms())
    return elements


def _has_foreign_element(smi, allowed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    return any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms())


def norm_invert(x, lo, hi):
    return 1 - np.clip((x - lo) / (hi - lo), 0, 1)


def norm_direct(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--train-csv", required=True, help="SurfPro-MD.csv (defines the element vocabulary)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=400, help="size of the small best-of-the-best list, pre element-filter")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    train_elements = get_training_elements(args.train_csv)
    print("training elements:", sorted(train_elements))

    df = pd.read_csv(args.scored)
    df = df[df["valid"]].reset_index(drop=True)
    print("n_valid:", len(df))

    global _allowed  # simple global for multiprocessing picklability
    _allowed = train_elements
    with Pool(processes=args.workers) as pool:
        foreign = pool.starmap(_has_foreign_element, [(s, train_elements) for s in df["smiles_std"]], chunksize=2000)
    df["has_foreign_element"] = foreign
    eligible = df[~df["has_foreign_element"]].copy()
    print(f"eligible (element-valid): {len(eligible)} / {len(df)}")

    p5_pcmc, p95_pcmc = eligible["pCMC_mean"].quantile([0.05, 0.95])
    p5_st, p95_st = eligible["surface_tension_avg_mean"].quantile([0.05, 0.95])

    # pCMC = -log10(CMC): HIGHER pCMC = LOWER CMC = more efficient surfactant,
    # so it's maximized (norm_direct), not minimized -- confirmed empirically
    # 2026-07-21 (dodecyl sulfate/SDS: pCMC=2.03 -> CMC=9.3mM, matching SDS's
    # known experimental CMC). SurfTen is still minimized (norm_invert).
    s1 = norm_direct(eligible["pCMC_mean"], p5_pcmc, p95_pcmc)
    s2 = norm_invert(eligible["surface_tension_avg_mean"], p5_st, p95_st)
    s3 = norm_invert(eligible["pCMC_std"], PCMC_UNC_MIN, PCMC_UNC_MAX)
    s4 = norm_invert(eligible["surface_tension_avg_std"], SURFTEN_UNC_MIN, SURFTEN_UNC_MAX)
    eligible["composite"] = (s1 * s2 * s3 * s4) ** 0.25
    eligible = eligible.sort_values("composite", ascending=False).reset_index(drop=True)

    best_path = f"{args.out_dir}/zinc_holdout_low_pCMC_low_SurfTen.csv"
    eligible.head(args.top_n).to_csv(best_path, index=False)
    print(f"top {args.top_n} -> {best_path}")

    n = len(eligible)
    for pct in [5, 10, 15, 20]:
        k = int(n * pct / 100)
        tier_path = f"{args.out_dir}/zinc_holdout_top{pct}pct.csv.gz"
        eligible.head(k).to_csv(tier_path, index=False, compression="gzip")
        print(f"top {pct}%: {k} molecules -> {tier_path}")


if __name__ == "__main__":
    main()
