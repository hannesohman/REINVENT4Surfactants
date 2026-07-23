#!/usr/bin/env python3
"""
Full evaluation-metric suite for a generative run:
  - Validity, Uniqueness, Novelty (standard MOSES-style generative metrics)
  - Internal Diversity (IntDiv = 1 - mean pairwise Tanimoto, on a subsample)
  - Fragment (BRICS) / Scaffold (Bemis-Murcko) distributional similarity to
    the training set -- "does this look like what we trained on"
  - Fragment / Scaffold distributional similarity to a cached ZINC reference
    sample -- "does this look like plausible chemistry in general" (purely
    structural; NOT a claim about surfactant quality -- see README)
  - SurfPro real-holdout rediscovery, broken down by quality tier (1=best,
    5=worst; ground truth, real measured properties)
  - ZINC rediscovery, broken down by quality tier (1=best, 5=worst;
    surrogate-scored, so a softer signal -- see README)

Run workflow/build_zinc_reference.py once first to produce the cached ZINC
tier files and reference profile this script reads.

The expensive fixed-cost inputs (training set fragment/scaffold profile,
ZINC tier membership sets, ZINC reference profile) are loadable once via
load_resources() and reused across many evaluate() calls -- see
workflow/run_replicated_eval.py, which evaluates several independent RL
replicates without re-reading the ~10M-line ZINC tier files each time.

Usage:
    python workflow/evaluate_run.py \
        --generated "runs/compare_1/optimized/rep_*/trial_1.csv" \
        --train-csv data/surfpro_expanded_trainval_only.csv \
        --surfpro-holdout data/surfpro_real_holdout_test_split.csv \
        --out results_optimized.json
"""
import argparse
import glob
import gzip
import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def cosine_sim_counters(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def fragment_and_scaffold_counts(mols):
    frag_counts, scaf_counts = Counter(), Counter()
    for mol in mols:
        try:
            frag_counts.update(BRICS.BRICSDecompose(mol))
        except Exception:
            pass
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaf:
                scaf_counts[scaf] += 1
        except Exception:
            pass
    return frag_counts, scaf_counts


def internal_diversity(mols, sample_n=2000, seed=42):
    rng = np.random.RandomState(seed)
    if len(mols) > sample_n:
        idx = rng.choice(len(mols), sample_n, replace=False)
        mols = [mols[i] for i in idx]
    fps = [morgan_fp(m) for m in mols]
    n = len(fps)
    if n < 2:
        return float("nan")
    sims = []
    for i in range(n - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return 1 - float(np.mean(sims)) if sims else float("nan")


def tier_aggregate(tier_hits: dict, tiers: list[int]) -> tuple[int, int]:
    hits = sum(tier_hits[t]["hits"] for t in tiers if t in tier_hits)
    n = sum(tier_hits[t]["n"] for t in tiers if t in tier_hits)
    return hits, n


@dataclass
class EvalResources:
    """Fixed-cost inputs, loaded once and reused across many evaluate() calls."""
    train_canon_set: set
    train_frag: Counter
    train_scaf: Counter
    surfpro_df: pd.DataFrame
    zinc_tier_sets: dict  # tier(int) -> set(smiles_std)
    zinc_frag: Counter
    zinc_scaf: Counter


def load_resources(train_csv, train_smiles_col, surfpro_holdout, zinc_quintile_dir, zinc_reference) -> EvalResources:
    train_df = pd.read_csv(train_csv)
    train_canon_set = set(train_df[train_smiles_col].apply(canon).dropna())
    train_mols = [m for m in (Chem.MolFromSmiles(s) for s in train_canon_set) if m is not None]
    train_frag, train_scaf = fragment_and_scaffold_counts(train_mols)

    surfpro_df = pd.read_csv(surfpro_holdout)
    surfpro_df["canon"] = surfpro_df["SMILES_canonical"].apply(canon)

    zinc_tier_sets = {}
    for tier in [1, 2, 3, 4, 5]:
        path = f"{zinc_quintile_dir}/zinc_quintile_tier{tier}.smi.gz"
        with gzip.open(path, "rt") as f:
            zinc_tier_sets[tier] = set(line.strip() for line in f)

    with gzip.open(zinc_reference, "rt") as f:
        zinc_ref = json.load(f)
    zinc_frag, zinc_scaf = Counter(zinc_ref["fragment_counts"]), Counter(zinc_ref["scaffold_counts"])

    return EvalResources(
        train_canon_set=train_canon_set, train_frag=train_frag, train_scaf=train_scaf,
        surfpro_df=surfpro_df, zinc_tier_sets=zinc_tier_sets,
        zinc_frag=zinc_frag, zinc_scaf=zinc_scaf,
    )


def evaluate(raw_df: pd.DataFrame, generated_smiles_col: str, resources: EvalResources, intdiv_sample: int = 2000) -> dict:
    n_total = len(raw_df)

    raw_df = raw_df.copy()
    raw_df["canon"] = raw_df[generated_smiles_col].apply(canon)
    n_valid = int(raw_df["canon"].notna().sum())
    validity = n_valid / n_total if n_total else float("nan")

    unique_canon = raw_df["canon"].dropna().unique().tolist()
    n_unique = len(unique_canon)
    uniqueness = n_unique / n_valid if n_valid else float("nan")

    n_novel = sum(1 for s in unique_canon if s not in resources.train_canon_set)
    novelty = n_novel / n_unique if n_unique else float("nan")

    unique_mols = [m for m in (Chem.MolFromSmiles(s) for s in unique_canon) if m is not None]
    intdiv = internal_diversity(unique_mols, sample_n=intdiv_sample)

    gen_frag, gen_scaf = fragment_and_scaffold_counts(unique_mols)
    frag_sim_train = cosine_sim_counters(gen_frag, resources.train_frag)
    scaf_sim_train = cosine_sim_counters(gen_scaf, resources.train_scaf)
    frag_sim_zinc = cosine_sim_counters(gen_frag, resources.zinc_frag)
    scaf_sim_zinc = cosine_sim_counters(gen_scaf, resources.zinc_scaf)

    unique_set = set(unique_canon)
    surfpro = resources.surfpro_df
    surfpro_tier_hits = {}
    for tier in sorted(surfpro["quality_tier"].unique()):
        tier_df = surfpro[surfpro["quality_tier"] == tier]
        hits = int(tier_df["canon"].isin(unique_set).sum())
        surfpro_tier_hits[int(tier)] = {"hits": hits, "n": len(tier_df), "rate": hits / len(tier_df)}
    top2_hits, top2_n = tier_aggregate(surfpro_tier_hits, [1, 2])
    bot2_hits, bot2_n = tier_aggregate(surfpro_tier_hits, [4, 5])

    zinc_tier_hits = {}
    for tier, tier_smiles in resources.zinc_tier_sets.items():
        hits = sum(1 for s in unique_canon if s in tier_smiles)
        zinc_tier_hits[tier] = {"hits": hits, "n": len(tier_smiles), "rate": hits / len(tier_smiles) if tier_smiles else 0.0}

    return {
        "n_generated_total": n_total,
        "n_valid": n_valid,
        "n_unique_valid": n_unique,
        "n_novel": n_novel,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "internal_diversity": intdiv,
        "frag_similarity_train": frag_sim_train,
        "scaf_similarity_train": scaf_sim_train,
        "frag_similarity_zinc": frag_sim_zinc,
        "scaf_similarity_zinc": scaf_sim_zinc,
        "surfpro_tier_hits": surfpro_tier_hits,
        "surfpro_top2_vs_bottom2": {
            "top2": {"hits": top2_hits, "n": top2_n, "rate": top2_hits / top2_n},
            "bottom2": {"hits": bot2_hits, "n": bot2_n, "rate": bot2_hits / bot2_n},
        },
        "zinc_tier_hits": zinc_tier_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="glob of CSV(s)")
    ap.add_argument("--generated-smiles-col", default="SMILES")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--train-smiles-col", default="SMILES_canonical")
    ap.add_argument("--surfpro-holdout", required=True)
    ap.add_argument("--zinc-quintile-dir", default="data")
    ap.add_argument("--zinc-reference", default="data/zinc_reference_profile.json.gz")
    ap.add_argument("--intdiv-sample", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.generated))
    if not paths:
        raise FileNotFoundError(f"no files matched: {args.generated}")
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"loaded {len(raw)} generated rows from {len(paths)} file(s)", flush=True)

    print("loading fixed-cost resources (train profile, ZINC tiers, ZINC reference)...", flush=True)
    resources = load_resources(
        args.train_csv, args.train_smiles_col, args.surfpro_holdout,
        args.zinc_quintile_dir, args.zinc_reference,
    )

    print("evaluating...", flush=True)
    results = evaluate(raw, args.generated_smiles_col, resources, intdiv_sample=args.intdiv_sample)

    print(f"validity={results['validity']:.4f}  uniqueness={results['uniqueness']:.4f}  novelty={results['novelty']:.4f}", flush=True)
    print(f"internal_diversity={results['internal_diversity']:.4f}", flush=True)
    print(f"frag_similarity_train={results['frag_similarity_train']:.4f}  scaf_similarity_train={results['scaf_similarity_train']:.4f}", flush=True)
    print(f"frag_similarity_zinc={results['frag_similarity_zinc']:.4f}  scaf_similarity_zinc={results['scaf_similarity_zinc']:.4f}", flush=True)
    print("surfpro_tier_hits:", results["surfpro_tier_hits"], flush=True)
    t2, b2 = results["surfpro_top2_vs_bottom2"]["top2"], results["surfpro_top2_vs_bottom2"]["bottom2"]
    print(f"top-2 tiers: {t2['hits']}/{t2['n']} ({100*t2['rate']:.1f}%)   "
          f"bottom-2 tiers: {b2['hits']}/{b2['n']} ({100*b2['rate']:.1f}%)", flush=True)
    print("zinc_tier_hits:", results["zinc_tier_hits"], flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
