#!/usr/bin/env python3
"""
Check whether the top-N% highest-scoring generated molecules from a run are
already known compounds in ChEMBL -- the same percentile convention as the
HPO objective in workflow/run_production_combo.py (top-5% of the pool by
Score), applied here as a "have we just rediscovered known chemistry"
sanity/novelty check against a large public bioactive-molecule database,
independent of the SurfPro/ZINC rediscovery checks elsewhere in this project.

Matching is done on InChIKey (via RDKit), not raw SMILES, since it is
robust to representation differences between REINVENT's and ChEMBL's
canonicalization. Both a strict match (full InChIKey -- same molecule,
including stereochemistry) and a looser skeleton match (first 14 characters
of the InChIKey -- same connectivity, ignoring stereochemistry/tautomer/salt
state) are reported, since generated SMILES carry no stereochemistry and many
ChEMBL entries are salts of the compound REINVENT would generate as the free
form.

Run workflow/build_chembl_reference.py once first to produce the cached
data/chembl_reference.json.gz this script reads.

Usage:
    python workflow/check_chembl_membership.py \
        --generated "runs/production/<combo>/production/rep_*/trial_1.csv" \
        --generated-smiles-col SMILES --score-col Score --top-pct 5 \
        --chembl-reference data/chembl_reference.json.gz
"""
import argparse
import glob
import gzip
import json

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")


def to_inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="glob of CSV(s) with generated SMILES + Score")
    ap.add_argument("--generated-smiles-col", default="SMILES")
    ap.add_argument("--score-col", default="Score")
    ap.add_argument("--top-pct", type=float, default=5.0, help="top-N%% by score to check (default 5)")
    ap.add_argument("--chembl-reference", default="data/chembl_reference.json.gz")
    ap.add_argument("--show-hits", type=int, default=20)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.generated))
    if not paths:
        raise FileNotFoundError(f"no files matched: {args.generated}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(paths)} file(s)")

    valid_mask = df[args.generated_smiles_col].apply(lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    df = df[valid_mask]
    unique_df = df.drop_duplicates(subset=[args.generated_smiles_col])
    n_top = max(1, round(len(unique_df) * args.top_pct / 100))
    top_df = unique_df.sort_values(args.score_col, ascending=False).head(n_top)
    print(f"unique valid molecules: {len(unique_df)}; checking top {args.top_pct}% = {n_top} molecules")

    top_df = top_df.copy()
    top_df["inchikey"] = top_df[args.generated_smiles_col].apply(to_inchikey)
    n_no_inchikey = top_df["inchikey"].isna().sum()
    if n_no_inchikey:
        print(f"warning: {n_no_inchikey} of the top {n_top} could not be converted to an InChIKey")
    top_df = top_df.dropna(subset=["inchikey"])
    top_df["skeleton"] = top_df["inchikey"].str[:14]

    print(f"loading ChEMBL reference from {args.chembl_reference} ...")
    with gzip.open(args.chembl_reference, "rt") as f:
        ref = json.load(f)
    full_set = set(ref["inchikeys_full"])
    skeleton_set = set(ref["inchikeys_skeleton"])
    id_map = ref["inchikey_to_chembl_id"]
    print(f"reference: {ref['n_compounds']} ChEMBL compounds, "
          f"{len(full_set)} unique InChIKeys, {len(skeleton_set)} unique skeletons")

    full_hits = top_df[top_df["inchikey"].isin(full_set)]
    skeleton_hits = top_df[top_df["skeleton"].isin(skeleton_set) & ~top_df["inchikey"].isin(full_set)]

    print(f"\n=== RESULTS (top {args.top_pct}% = {len(top_df)} molecules with a valid InChIKey) ===")
    print(f"Exact ChEMBL match (full InChIKey):     {len(full_hits)}  ({100*len(full_hits)/len(top_df):.2f}%)")
    print(f"Skeleton-only match (stereo/salt differs): {len(skeleton_hits)}  ({100*len(skeleton_hits)/len(top_df):.2f}%)")
    print(f"No match in ChEMBL:                      {len(top_df) - len(full_hits) - len(skeleton_hits)}")

    if len(full_hits):
        print(f"\nExact matches (showing up to {args.show_hits}):")
        for _, row in full_hits.head(args.show_hits).iterrows():
            chembl_id = id_map.get(row["inchikey"], "?")
            print(f"  {row[args.generated_smiles_col]}  ->  {chembl_id}  (score={row[args.score_col]:.4f})")


if __name__ == "__main__":
    main()
