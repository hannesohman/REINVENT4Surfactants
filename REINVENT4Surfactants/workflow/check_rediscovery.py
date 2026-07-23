#!/usr/bin/env python3
"""
Check exact-canonical-SMILES overlap between a set of generated/sampled
molecules and one or more holdout files -- the "did we (re)discover any of
these" check used throughout this project's validation work.

Usage:
    # RL staged-learning output (3 multiple_*_1.csv files) vs the 4 staggered ZINC tiers
    python workflow/check_rediscovery.py \
        --generated "runs/test/<run_id>/generation_0/combo_*/multiple_*/multiple_*_1.csv" \
        --generated-smiles-col SMILES \
        --holdout data/zinc_holdout_top5pct.csv.gz:smiles_std \
        --holdout data/zinc_holdout_top10pct.csv.gz:smiles_std \
        --holdout data/zinc_holdout_top15pct.csv.gz:smiles_std \
        --holdout data/zinc_holdout_top20pct.csv.gz:smiles_std

    # plain `sampling` run_type output vs the real, ground-truth holdout
    python workflow/check_rediscovery.py \
        --generated runs/baselines/prior_samples.csv --generated-smiles-col SMILES \
        --holdout data/surfpro_real_holdout_test_split.csv:SMILES_canonical

Each --holdout is "path:column_name" (column defaults to "smiles_std" if omitted).
--generated accepts a glob; all matches are concatenated.
"""
import argparse
import glob

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="glob of CSV(s) with generated SMILES")
    ap.add_argument("--generated-smiles-col", default="SMILES")
    ap.add_argument("--holdout", action="append", required=True, help="path[:smiles_column], repeatable")
    ap.add_argument("--show-hits", type=int, default=10, help="how many example hit SMILES to print per holdout")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.generated))
    if not paths:
        raise FileNotFoundError(f"no files matched: {args.generated}")
    gen = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"loaded {len(gen)} rows from {len(paths)} file(s)")

    gen["canon"] = gen[args.generated_smiles_col].apply(canon)
    gen_unique = set(gen["canon"].dropna().unique())
    print(f"unique valid generated molecules: {len(gen_unique)}\n")

    for spec in args.holdout:
        if ":" in spec:
            path, col = spec.rsplit(":", 1)
        else:
            path, col = spec, "smiles_std"

        holdout = pd.read_csv(path, usecols=[col])
        holdout["canon"] = holdout[col].apply(canon)
        holdout_set = set(holdout["canon"].dropna())

        hits = gen_unique & holdout_set
        pct_of_holdout = 100 * len(hits) / len(holdout_set) if holdout_set else 0
        pct_of_gen = 100 * len(hits) / len(gen_unique) if gen_unique else 0
        print(f"{path}  ({len(holdout_set)} molecules)")
        print(f"  hits: {len(hits)}  ({pct_of_holdout:.4f}% of holdout, {pct_of_gen:.4f}% of generated)")
        for h in list(hits)[: args.show_hits]:
            print(f"    - {h}")
        print()


if __name__ == "__main__":
    main()
