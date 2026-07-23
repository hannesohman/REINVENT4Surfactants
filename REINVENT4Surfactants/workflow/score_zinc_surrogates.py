#!/usr/bin/env python3
"""
Score the ZINC in-stock pull (data/ZINC/zinc_instock_combined.csv.gz) against
the full 9-property SurfPro-MD surrogate ensemble (models.pkl), for use as a
"re-discovery" holdout set: molecules picked afterwards by surrogate-predicted
property, not by any structural surfactant-likeness filter.

Preprocessing mirrors surfactant-surrogates/SurfPro-MD/MD-simulations/
extract_data_canonicalised.py's standardize_smiles: keep the largest fragment
(strip counterions/salts), Cleanup, FragmentParent, Uncharger (balance
charges), canonicalize -- then re-parse the canonical SMILES before computing
RDKit descriptors, matching how models.pkl's training features were built.

Chunked + resumable: each chunk of the input is scored and written to its own
gzip CSV shard under <output-dir>/chunks/, with completed chunk indices
recorded in a manifest so a re-run skips already-finished chunks.
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")

_DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)


def standardize_and_featurize(smiles: str):
    """Returns (canonical_smiles, feature_dict) or (None, None) on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        mol2 = Chem.MolFromSmiles(canonical)
        if mol2 is None:
            return None, None
        values = _CALC.CalcDescriptors(mol2)
        feats = {f"rdkit-{n}": v for n, v in zip(_DESCRIPTOR_NAMES, values)}
        return canonical, feats
    except Exception:
        return None, None


def _worker(args):
    zinc_id, smiles = args
    canonical, feats = standardize_and_featurize(smiles)
    return zinc_id, canonical, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="zinc_instock_combined.csv.gz")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models-path", required=True)
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    chunks_dir = os.path.join(args.output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "completed_chunks.txt")

    completed = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            completed = {int(l.strip()) for l in f if l.strip()}
    print(f"[score_zinc] {len(completed)} chunks already completed, resuming", file=sys.stderr)

    print(f"[score_zinc] loading models from {args.models_path} ...", file=sys.stderr)
    t0 = time.time()
    with open(args.models_path, "rb") as f:
        all_models = pickle.load(f)
    targets = list(all_models.keys())
    feature_schema = next(iter(all_models[targets[0]].values()))["feature_schema"]
    target_fold_models = {
        t: [fm for split in all_models[t].values() for fm in split["fold_models"]]
        for t in targets
    }
    print(f"[score_zinc] loaded {len(targets)} targets in {time.time()-t0:.1f}s: {targets}", file=sys.stderr)

    reader = pd.read_csv(args.input, chunksize=args.chunk_size)
    pool = Pool(processes=args.workers)

    for chunk_idx, chunk in enumerate(reader):
        if chunk_idx in completed:
            continue

        t0 = time.time()
        results = pool.map(_worker, list(zip(chunk["zinc_id"], chunk["smiles"])), chunksize=200)
        zinc_ids, canonical_smiles, feat_dicts = zip(*results)

        valid_mask = np.array([f is not None for f in feat_dicts])
        feat_rows = []
        for f in feat_dicts:
            if f is None:
                feat_rows.append([np.nan] * len(feature_schema))
            else:
                feat_rows.append([f.get(name, np.nan) for name in feature_schema])
        X = np.array(feat_rows, dtype=np.float64)

        n_nan_per_row = np.isnan(X).sum(axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=2e9, neginf=-2e9)
        X = np.clip(X, -2e9, 2e9)
        t1 = time.time()

        out = pd.DataFrame({
            "zinc_id": zinc_ids,
            "smiles_std": canonical_smiles,
            "smiles_orig": chunk["smiles"].to_numpy(),
            "tranche": chunk["tranche"].to_numpy(),
            "reactivity": chunk["reactivity"].to_numpy(),
            "purchasability": chunk["purchasability"].to_numpy(),
            "valid": valid_mask,
            "n_nan_descriptors": n_nan_per_row,
        })

        for target in targets:
            fold_models = target_fold_models[target]
            preds = np.stack(
                [fm["model"].inplace_predict(fm["scaler"].transform(X)) for fm in fold_models],
                axis=0,
            )
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            mean[~valid_mask] = np.nan
            std[~valid_mask] = np.nan
            out[f"{target}_mean"] = mean
            out[f"{target}_std"] = std
        t2 = time.time()

        out_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}.csv.gz")
        out.to_csv(out_path, index=False, compression="gzip")

        with open(manifest_path, "a") as f:
            f.write(f"{chunk_idx}\n")

        n_invalid = (~valid_mask).sum()
        print(
            f"[score_zinc] chunk {chunk_idx}: {len(out)} rows "
            f"({n_invalid} invalid), featurize={t1-t0:.1f}s, infer={t2-t1:.1f}s, "
            f"wrote {out_path}",
            file=sys.stderr,
        )

    pool.close()
    pool.join()
    print("[score_zinc] all chunks done", file=sys.stderr)


if __name__ == "__main__":
    main()
