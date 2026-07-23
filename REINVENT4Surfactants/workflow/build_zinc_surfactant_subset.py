#!/usr/bin/env python3
"""
Extract a "surfactant-like" subset of ZINC by nearest-neighbor distance to
SurfPro-MD in normalized 9-property space, rather than by surrogate-score
ranking (that's what build_zinc_holdouts.py / build_zinc_reference.py do).

This answers a different question than the existing holdout builders: not
"which ZINC molecules score best on pCMC/SurfTen" but "which ZINC molecules
actually resemble the real, measured SurfPro-MD surfactants across all 9
properties" -- a property-space applicability-domain filter.

Method
------
1. SurfPro-MD.csv (measured) and zinc_scored_9props.csv.gz (surrogate-
   predicted `*_mean`) are combined and z-scored per property using pooled
   mean/std across *both* populations (per-property normalization requested
   over the unified set, not either alone).

2. A bounding box or single-ellipsoid (Mahalanobis) fit was deliberately
   NOT used: SurfPro-MD's 9 properties are correlated and the real data
   spans 6 distinct surf_types (anionic/cationic/gemini cationic/non-ionic/
   sugar-based non-ionic/zwitterionic) that don't form one blob, so a box or
   single ellipsoid would hugely overestimate the "surfactant-like" region.
   Instead: for every ZINC molecule, compute its distance to the *nearest*
   individual SurfPro-MD molecule (1-NN) in normalized property space. This
   adapts to whatever shape/multi-modality the real data has.

3. SurfPro-MD has real missingness -- only 567/1436 molecules have all 9
   properties measured (10 distinct missingness patterns; see module-level
   PATTERN note below). Restricting to complete cases would drop every
   zwitterionic surfactant and most anionics. Instead, distance from a ZINC
   candidate to a given SurfPro molecule uses *only the properties that
   SurfPro molecule has measured* (masked/per-pattern distance), normalized
   as an RMS-per-dimension distance so patterns using different numbers of
   properties stay comparable. ZINC's surrogate predictions are always
   complete (no masking needed on that side).

4. The "in-domain" distance threshold is calibrated from SurfPro-MD itself:
   its own leave-one-out nearest-neighbor distance distribution (how far
   apart real surfactants typically sit from each other) sets the natural
   scale, rather than an arbitrary cutoff.

Usage:
    python workflow/build_zinc_surfactant_subset.py \\
        --scored data/ZINC/zinc_scored_9props.csv.gz \\
        --surfpro-csv /path/to/SurfPro-MD/SurfPro-MD.csv \\
        --out-dir data \\
        --workers 48
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.neighbors import BallTree

RDLogger.DisableLog("rdApp.*")

PROPS = [
    "pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "pC20",
    "D_MOL", "D_SOL", "surface_tension_avg", "viscosity",
]
ZINC_MEAN_COLS = [f"{p}_mean" for p in PROPS]


def get_training_elements(surfpro_csv: str) -> set[str]:
    df = pd.read_csv(surfpro_csv, usecols=["SMILES_canonical"])
    elements = set()
    for smi in df["SMILES_canonical"].dropna().unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        elements.update(atom.GetSymbol() for atom in mol.GetAtoms())
    return elements


def _has_foreign_element(smi, allowed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    return any(a.GetSymbol() not in allowed for a in mol.GetAtoms())


def masked_rms_nn_distance(query, targets_by_pattern):
    """
    query: (N, 9) array, always fully populated (ZINC).
    targets_by_pattern: list of (cols_idx, target_matrix) where target_matrix
        is (m, len(cols_idx)), z-scored, one entry per distinct SurfPro
        missingness pattern.

    Returns (dist, nn_row_global_idx) -- for each query row, the smallest
    RMS-per-dimension distance to any target across all patterns, and which
    global SurfPro row index achieved it.
    """
    n = query.shape[0]
    best_dist = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int64)
    for cols_idx, target_matrix, global_idx in targets_by_pattern:
        if target_matrix.shape[0] == 0:
            continue
        sub_query = query[:, cols_idx]
        tree = BallTree(target_matrix, metric="euclidean")
        dist, ind = tree.query(sub_query, k=1)
        dist = dist[:, 0] / np.sqrt(len(cols_idx))  # RMS per dimension
        ind = ind[:, 0]
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_idx[better] = global_idx[ind[better]]
    return best_dist, best_idx


def self_nn_distance(z, present_mask):
    """
    Leave-one-out nearest-neighbor distance among SurfPro-MD rows themselves,
    using the same masked/RMS scheme: for row i, only compare against rows j
    that have *at least* all of row i's present properties, using row i's
    property set. Brute-force (N~1436, trivial).
    """
    n = z.shape[0]
    best = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        cols = np.where(present_mask[i])[0]
        if len(cols) == 0:
            continue
        candidates = np.where(present_mask[:, cols].all(axis=1))[0]
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        diff = z[np.ix_(candidates, cols)] - z[i, cols]
        d = np.sqrt((diff ** 2).mean(axis=1))
        j = np.argmin(d)
        best[i] = d[j]
        best_idx[i] = candidates[j]
    return best, best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="zinc_scored_9props.csv.gz")
    ap.add_argument("--surfpro-csv", required=True, help="SurfPro-MD.csv (measured properties)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument(
        "--threshold-percentile", type=float, default=95.0,
        help="Percentile of SurfPro-MD's own leave-one-out NN-distance distribution used as the in-domain cutoff.",
    )
    args = ap.parse_args()

    print("loading SurfPro-MD...", flush=True)
    sp = pd.read_csv(args.surfpro_csv)
    sp = sp.dropna(subset=["SMILES_canonical"]).reset_index(drop=True)
    sp_vals = sp[PROPS].to_numpy(dtype=float)
    present = ~np.isnan(sp_vals)
    print(f"SurfPro-MD: {len(sp)} molecules, "
          f"{present.all(axis=1).sum()} with all 9 properties", flush=True)

    print("loading scored ZINC pool...", flush=True)
    zinc = pd.read_csv(args.scored)
    zinc = zinc[zinc["valid"]].reset_index(drop=True)
    zinc_vals = zinc[ZINC_MEAN_COLS].to_numpy(dtype=float)
    zinc_valid_rows = ~np.isnan(zinc_vals).any(axis=1)
    if not zinc_valid_rows.all():
        print(f"dropping {(~zinc_valid_rows).sum()} ZINC rows with NaN predictions", flush=True)
        zinc = zinc[zinc_valid_rows].reset_index(drop=True)
        zinc_vals = zinc_vals[zinc_valid_rows]
    print(f"ZINC valid pool: {len(zinc)}", flush=True)

    print("filtering ZINC to SurfPro-MD's element vocabulary...", flush=True)
    train_elements = get_training_elements(args.surfpro_csv)
    print("training elements:", sorted(train_elements), flush=True)
    with Pool(processes=args.workers) as pool:
        foreign = pool.starmap(
            _has_foreign_element,
            [(s, train_elements) for s in zinc["smiles_std"]],
            chunksize=2000,
        )
    zinc = zinc[~np.array(foreign)].reset_index(drop=True)
    zinc_vals = zinc[ZINC_MEAN_COLS].to_numpy(dtype=float)
    print(f"element-eligible ZINC pool: {len(zinc)}", flush=True)

    # ---- pooled normalization over SurfPro (measured) + ZINC (predicted) ----
    print("computing pooled per-property normalization stats...", flush=True)
    pooled_mean = np.full(len(PROPS), np.nan)
    pooled_std = np.full(len(PROPS), np.nan)
    for j in range(len(PROPS)):
        pooled = np.concatenate([sp_vals[present[:, j], j], zinc_vals[:, j]])
        pooled_mean[j] = pooled.mean()
        pooled_std[j] = pooled.std()
    stats_df = pd.DataFrame({"property": PROPS, "pooled_mean": pooled_mean, "pooled_std": pooled_std})
    stats_df.to_csv(f"{args.out_dir}/zinc_surfactant_subset_normalization.csv", index=False)
    print(stats_df, flush=True)

    sp_z = (sp_vals - pooled_mean) / pooled_std  # NaNs preserved where not present
    zinc_z = (zinc_vals - pooled_mean) / pooled_std  # fully dense

    # ---- group SurfPro rows by exact missingness pattern ----
    patterns = [tuple(row) for row in present]
    unique_patterns = sorted(set(patterns), key=lambda p: -sum(p))
    print(f"{len(unique_patterns)} distinct SurfPro-MD missingness patterns", flush=True)

    targets_by_pattern = []
    for pat in unique_patterns:
        cols_idx = np.array([i for i, present_i in enumerate(pat) if present_i])
        if len(cols_idx) == 0:
            continue
        row_mask = np.array([p == pat for p in patterns])
        global_idx = np.where(row_mask)[0]
        target_matrix = sp_z[np.ix_(global_idx, cols_idx)]
        targets_by_pattern.append((cols_idx, target_matrix, global_idx))
        print(f"  pattern with {len(cols_idx)} properties: {row_mask.sum()} SurfPro molecules "
              f"({[PROPS[c] for c in cols_idx]})", flush=True)

    # ---- calibrate threshold from SurfPro-MD's own leave-one-out NN distances ----
    print("calibrating threshold from SurfPro-MD self nearest-neighbor distances...", flush=True)
    self_dist, self_nn_idx = self_nn_distance(sp_z, present)
    finite = np.isfinite(self_dist)
    threshold = np.percentile(self_dist[finite], args.threshold_percentile)
    print(f"self-NN distance: n={finite.sum()}, median={np.median(self_dist[finite]):.4f}, "
          f"p{args.threshold_percentile:.0f}={threshold:.4f}, max={self_dist[finite].max():.4f}", flush=True)

    sp_self = sp.loc[finite, ["SMILES_canonical", "surf_type"]].copy() if "surf_type" in sp.columns else sp.loc[finite, ["SMILES_canonical"]].copy()
    sp_self["self_nn_distance"] = self_dist[finite]
    sp_self.to_csv(f"{args.out_dir}/surfpro_self_nn_distances.csv", index=False)

    # ---- distance from every ZINC candidate to nearest SurfPro-MD molecule ----
    print("computing ZINC -> SurfPro-MD nearest-neighbor distances...", flush=True)
    dist, nn_idx = masked_rms_nn_distance(zinc_z, targets_by_pattern)

    out = zinc[["zinc_id", "smiles_std", "tranche", "purchasability"] + ZINC_MEAN_COLS].copy()
    out["nn_distance"] = dist
    out["nn_surfpro_smiles"] = sp.loc[nn_idx.clip(min=0), "SMILES_canonical"].to_numpy()
    if "surf_type" in sp.columns:
        out["nn_surfpro_surf_type"] = sp.loc[nn_idx.clip(min=0), "surf_type"].to_numpy()
    out["in_domain"] = out["nn_distance"] <= threshold

    out = out.sort_values("nn_distance").reset_index(drop=True)
    full_path = f"{args.out_dir}/zinc_surfactant_subset_all_distances.csv.gz"
    out.to_csv(full_path, index=False, compression="gzip")
    print(f"wrote all {len(out)} scored candidates -> {full_path}", flush=True)

    subset = out[out["in_domain"]].reset_index(drop=True)
    subset_path = f"{args.out_dir}/zinc_surfactant_subset.csv.gz"
    subset.to_csv(subset_path, index=False, compression="gzip")
    print(f"in-domain subset (nn_distance <= {threshold:.4f}, "
          f"p{args.threshold_percentile:.0f} of SurfPro-MD self-NN distances): "
          f"{len(subset)} / {len(out)} ({100 * len(subset) / len(out):.3f}%) -> {subset_path}", flush=True)


if __name__ == "__main__":
    main()
