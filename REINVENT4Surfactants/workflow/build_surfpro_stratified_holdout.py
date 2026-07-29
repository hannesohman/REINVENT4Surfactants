#!/usr/bin/env python3
"""
Build a SurfPro-MD train/holdout split, ranked by a composite of real
measured pCMC and SurfTen. Two modes:

  --mode stratified (default): 26 molecules per quintile tier => 130 holdout,
      1421 trainval (the original design, used for the tier-discrimination
      analysis in the README's Findings).
  --mode top_n: flat top-N holdout (highest composite score), rest trainval
      (used for the 2026-07-27 production runs' simpler top-100 holdout).

pCMC is -log10(CMC in mol/L) (pH-style; confirmed empirically 2026-07-21 --
e.g. dodecyl sulfate/SDS has pCMC=2.03 in SurfPro-MD.csv, implying
CMC=9.3 mM, matching SDS's known experimental CMC). HIGHER pCMC means LOWER
CMC, i.e. a MORE efficient surfactant -- so pCMC should be MAXIMIZED, not
minimized. An earlier version of this composite (and of config.json's RL
objective) treated pCMC as "lower is better" like SurfTen, which is backwards.
This script (and the fixed config.json/build_zinc_*.py) correct that.

  --mode cluster: Butina-cluster the full dataset on Morgan fingerprints
      (so homologous series -- same scaffold, different alkyl chain length --
      land in the same cluster), take the mean composite score per cluster,
      and greedily add whole clusters (best mean-composite first) to the
      holdout until the target size is reached. Unlike top_n, this never
      splits a homolog family across train/holdout, which top_n did and
      which let transfer learning implicitly "see" 81% of the ring-containing
      holdout molecules' exact scaffolds via a different chain length (found
      2026-07-29: pure TL-checkpoint sampling, no RL at all, rediscovered 65%
      of the top_n holdout -- see README).

Usage:
    python workflow/build_surfpro_stratified_holdout.py \
        --input data/surfpro_expanded.csv \
        --holdout-out data/surfpro_real_holdout_test_split.csv \
        --trainval-out data/surfpro_expanded_trainval_only.csv \
        --mode stratified --n-per-tier 26 --seed 42

    python workflow/build_surfpro_stratified_holdout.py \
        --mode top_n --top-n 100

    python workflow/build_surfpro_stratified_holdout.py \
        --mode cluster --cluster-cutoff 0.35 --target-min 100 --target-max 120
"""
import argparse

import numpy as np
import pandas as pd


def norm_high_is_good(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def norm_low_is_good(x, lo, hi):
    return 1 - np.clip((x - lo) / (hi - lo), 0, 1)


def butina_clusters(smiles_list, cutoff, radius=2, nbits=2048):
    """Cluster molecules so that homologous series (same core scaffold,
    different alkyl chain length) always land in one cluster. Pure Tanimoto/
    Butina clustering on Morgan fingerprints misses this for large chain-
    length deltas (the fingerprint distance grows with the chain-length
    difference even though the Murcko scaffold is identical) -- found
    2026-07-29 while fixing the flat top-100 holdout's homolog leakage (see
    README). So this unions molecules that are EITHER fingerprint-similar
    (Tanimoto distance < cutoff) OR share an identical non-empty Murcko
    scaffold, via union-find; connected components are the final clusters.
    """
    from collections import defaultdict

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols]
    scaffolds = []
    for m in mols:
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            scaffolds.append("")

    n = len(mols)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    scaf_groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        if s:
            scaf_groups[s].append(i)
    for idxs in scaf_groups.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for j, sim in enumerate(sims):
            if 1 - sim < cutoff:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def compute_composite(df: pd.DataFrame) -> pd.Series:
    p5_pcmc, p95_pcmc = df["pCMC"].quantile([0.05, 0.95])
    p5_st, p95_st = df["SurfTen"].quantile([0.05, 0.95])
    print(f"pCMC p5/p95: {p5_pcmc:.4f} / {p95_pcmc:.4f}", flush=True)
    print(f"SurfTen p5/p95: {p5_st:.4f} / {p95_st:.4f}", flush=True)

    s_pcmc = norm_high_is_good(df["pCMC"], p5_pcmc, p95_pcmc)   # HIGH pCMC = good (low CMC)
    s_surften = norm_low_is_good(df["SurfTen"], p5_st, p95_st)  # LOW SurfTen = good
    return (s_pcmc * s_surften) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/surfpro_expanded.csv")
    ap.add_argument("--holdout-out", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--trainval-out", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--mode", choices=["stratified", "top_n", "cluster"], default="stratified")
    ap.add_argument("--n-per-tier", type=int, default=26, help="stratified mode only")
    ap.add_argument("--top-n", type=int, default=100, help="top_n mode only")
    ap.add_argument("--cluster-cutoff", type=float, default=0.35, help="cluster mode only: Butina Tanimoto-distance cutoff")
    ap.add_argument("--target-min", type=int, default=100, help="cluster mode only")
    ap.add_argument("--target-max", type=int, default=120, help="cluster mode only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"loaded {len(df)} molecules from {args.input}", flush=True)

    df["true_composite"] = compute_composite(df)

    if args.mode == "stratified":
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
    elif args.mode == "top_n":
        df = df.sort_values("true_composite", ascending=False).reset_index(drop=True)
        holdout = df.head(args.top_n).copy()
        trainval = df.iloc[args.top_n:].copy()

        holdout.to_csv(args.holdout_out, index=False)
        trainval.to_csv(args.trainval_out, index=False)
        print(f"holdout (top {args.top_n}): {len(holdout)} -> {args.holdout_out}", flush=True)
        print(f"trainval: {len(trainval)} -> {args.trainval_out}", flush=True)
        print(
            f"holdout composite range: {holdout['true_composite'].min():.4f} - "
            f"{holdout['true_composite'].max():.4f}", flush=True,
        )
    else:  # cluster
        df = df.reset_index(drop=True)
        print(f"clustering {len(df)} molecules (Butina, cutoff={args.cluster_cutoff})...", flush=True)
        clusters = butina_clusters(df["SMILES_canonical"].tolist(), args.cluster_cutoff)
        print(f"n clusters: {len(clusters)} (sizes: min={min(len(c) for c in clusters)}, "
              f"max={max(len(c) for c in clusters)}, "
              f"median={sorted(len(c) for c in clusters)[len(clusters)//2]})", flush=True)

        cluster_mean = [(df.loc[list(c), "true_composite"].mean(), c) for c in clusters]
        cluster_mean.sort(key=lambda x: x[0], reverse=True)

        holdout_idx = []
        for mean_score, members in cluster_mean:
            if len(holdout_idx) >= args.target_min:
                break
            holdout_idx.extend(members)
            print(f"  + cluster (n={len(members)}, mean_composite={mean_score:.4f}) "
                  f"-> running total {len(holdout_idx)}", flush=True)
            if len(holdout_idx) > args.target_max:
                print(f"  [note: overshot target-max={args.target_max}, "
                      f"but this cluster was needed to clear target-min={args.target_min}]", flush=True)

        holdout = df.loc[holdout_idx].copy()
        trainval = df.drop(holdout.index).copy()

        holdout.to_csv(args.holdout_out, index=False)
        trainval.to_csv(args.trainval_out, index=False)
        print(f"\nholdout (cluster-based): {len(holdout)} -> {args.holdout_out}", flush=True)
        print(f"trainval: {len(trainval)} -> {args.trainval_out}", flush=True)
        print(
            f"holdout composite range: {holdout['true_composite'].min():.4f} - "
            f"{holdout['true_composite'].max():.4f}", flush=True,
        )


if __name__ == "__main__":
    main()
