#!/usr/bin/env python3
"""
One-time (cached) prep for the "similarity/rediscovery vs. ZINC" metrics used
by workflow/evaluate_run.py. Two independent outputs, deliberately built from
two different populations:

1. ZINC quintile tiers (data/zinc_quintile_tier{1-5}.smi.gz) -- for the
   rediscovery-rate metric. Same methodology as the SurfPro quintiles and the
   earlier ZINC top-N% holdouts: element-eligible (matches SurfPro-MD's
   training vocabulary), ranked by the equal-weighted composite score
   (pCMC/SurfTen/pCMC_Uncertainty/SurfTen_Uncertainty), split into 5 equal
   bins. Tier 1 = best (lowest pCMC+SurfTen, most reliable), tier 5 = worst.

2. ZINC reference profile (data/zinc_reference_profile.json.gz) -- for the
   fragment/scaffold *distributional* similarity metric, answering "does this
   look like real, plausible chemistry" rather than "is this a good
   surfactant". This is a uniform random sample of the full valid ZINC pool,
   NOT element-restricted and NOT property-ranked -- it should represent
   generic ZINC chemistry, not the surfactant-like slice of it. Fragment
   (BRICS) and scaffold (Bemis-Murcko) frequency distributions are
   precomputed and cached so evaluate_run.py never needs to touch the raw
   11M-molecule scored dataset.

Usage:
    python workflow/build_zinc_reference.py \
        --scored data/ZINC/zinc_scored_9props.csv.gz \
        --train-csv /path/to/SurfPro-MD/SurfPro-MD.csv \
        --out-dir data --reference-n 200000
"""
import argparse
import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def norm_invert(x, lo, hi):
    return 1 - np.clip((x - lo) / (hi - lo), 0, 1)


def norm_direct(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _has_foreign_element(smi, allowed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    return any(a.GetSymbol() not in allowed for a in mol.GetAtoms())


def _fragment_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    frags = None
    scaf = None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        pass
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        pass
    return frags, scaf


def fragment_profile(smiles_list, workers=48):
    from multiprocessing import Pool

    frag_counts = Counter()
    scaf_counts = Counter()
    n_ok = 0
    with Pool(processes=workers) as pool:
        for frags, scaf in pool.imap(_fragment_one, smiles_list, chunksize=500):
            if frags is None and scaf is None:
                continue
            n_ok += 1
            if frags:
                frag_counts.update(frags)
            if scaf:
                scaf_counts[scaf] += 1
    return frag_counts, scaf_counts, n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference-n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading scored ZINC dataset...", flush=True)
    df = pd.read_csv(args.scored)
    df = df[df["valid"]].reset_index(drop=True)
    print(f"n_valid: {len(df)}", flush=True)

    # ---- 1. ZINC quintile tiers (element-eligible, composite-ranked) ----
    train_elements = get_training_elements(args.train_csv)
    print("training elements:", sorted(train_elements), flush=True)

    print("checking element eligibility...", flush=True)
    from multiprocessing import Pool
    with Pool(processes=48) as pool:
        foreign = pool.starmap(
            _has_foreign_element, [(s, train_elements) for s in df["smiles_std"]], chunksize=2000
        )
    eligible = df[~np.array(foreign)].copy()
    print(f"eligible: {len(eligible)} / {len(df)}", flush=True)

    p5_pcmc, p95_pcmc = eligible["pCMC_mean"].quantile([0.05, 0.95])
    p5_st, p95_st = eligible["surface_tension_avg_mean"].quantile([0.05, 0.95])
    # pCMC = -log10(CMC): HIGHER pCMC = LOWER CMC = more efficient surfactant,
    # so it's maximized (norm_direct), not minimized -- confirmed empirically
    # 2026-07-21 (see README). SurfTen below is still minimized (norm_invert).
    s1 = norm_direct(eligible["pCMC_mean"], p5_pcmc, p95_pcmc)
    s2 = norm_invert(eligible["surface_tension_avg_mean"], p5_st, p95_st)
    s3 = norm_invert(eligible["pCMC_std"], PCMC_UNC_MIN, PCMC_UNC_MAX)
    s4 = norm_invert(eligible["surface_tension_avg_std"], SURFTEN_UNC_MIN, SURFTEN_UNC_MAX)
    eligible["composite"] = (s1 * s2 * s3 * s4) ** 0.25

    eligible["quintile"] = pd.qcut(
        eligible["composite"].rank(ascending=False, method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    print(eligible.groupby("quintile", observed=True).size(), flush=True)

    for tier in [1, 2, 3, 4, 5]:
        tier_smiles = eligible.loc[eligible["quintile"] == tier, "smiles_std"]
        out_path = f"{args.out_dir}/zinc_quintile_tier{tier}.smi.gz"
        with gzip.open(out_path, "wt") as f:
            for smi in tier_smiles:
                f.write(smi + "\n")
        print(f"tier {tier}: {len(tier_smiles)} -> {out_path}", flush=True)

    # ---- 2. ZINC reference profile (full pool, unrestricted, random sample) ----
    print(f"\nsampling {args.reference_n} molecules for the reference profile...", flush=True)
    ref_sample = df.sample(n=min(args.reference_n, len(df)), random_state=args.seed)
    ref_smiles_path = f"{args.out_dir}/zinc_reference_sample.smi.gz"
    with gzip.open(ref_smiles_path, "wt") as f:
        for smi in ref_sample["smiles_std"]:
            f.write(smi + "\n")
    print(f"saved raw reference sample -> {ref_smiles_path}", flush=True)

    print("computing BRICS fragment + Murcko scaffold profiles (this is the slow part)...", flush=True)
    frag_counts, scaf_counts, n_ok = fragment_profile(ref_sample["smiles_std"].tolist(), workers=48)
    print(f"fragmented {n_ok} / {len(ref_sample)} reference molecules", flush=True)
    print(f"n distinct fragments: {len(frag_counts)}, n distinct scaffolds: {len(scaf_counts)}", flush=True)

    profile = {
        "n_reference": n_ok,
        "fragment_counts": dict(frag_counts),
        "scaffold_counts": dict(scaf_counts),
    }
    profile_path = f"{args.out_dir}/zinc_reference_profile.json.gz"
    with gzip.open(profile_path, "wt") as f:
        json.dump(profile, f)
    print(f"saved reference profile -> {profile_path}", flush=True)


if __name__ == "__main__":
    main()
