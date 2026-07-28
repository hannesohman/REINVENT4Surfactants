#!/usr/bin/env python3
"""
Run REINVENT RL (3 replicates x N steps each, matching config.json's default
N_MULTIPLES_PER_GEN/N_STEPS_PER_MULT convention) under two hyperparameter sets
-- "default" (config.json's current SIGMA/LEARNING_RATE/BATCH_SIZE) and
"optimized" (the Optuna sweep's best trial) -- from the SAME fixed TL
checkpoint, then report rediscovery rate against the ZINC and SurfPro holdouts
for each.

Usage:
    python workflow/compare_hyperparams.py \
        --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
        --out-dir runs/compare_1 --replicates 3 --steps 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "workflow")
from optuna_rl_search import make_toml, REINVENT_PYTHON

RDLogger.DisableLog("rdApp.*")

CONFIGS = {
    "default": {"sigma": 120, "learning_rate": 0.00038716608106033025, "batch_size": 256},
    "optimized": {"sigma": 225, "learning_rate": 9.788453643515026e-05, "batch_size": 512},
}


def canon(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def run_replicate(config_name, params, replicate_idx, tl_model, prior_file, steps, out_dir):
    rep_dir = Path(out_dir) / config_name / f"rep_{replicate_idx}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "checkpoints").mkdir(exist_ok=True)
    (rep_dir / "tb_logdir").mkdir(exist_ok=True)

    toml_text = make_toml(
        rep_dir, tl_model, prior_file, params["sigma"], params["learning_rate"],
        params["batch_size"], steps, seed=replicate_idx,
    )
    toml_path = rep_dir / "rep.toml"
    toml_path.write_text(toml_text)

    log_path = rep_dir / "rep.log"
    result = subprocess.run(
        [REINVENT_PYTHON, "-u", "workflow/reinvent_with_lm.py", "-l", str(log_path), str(toml_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[{config_name} rep {replicate_idx}] FAILED\n{result.stderr[-4000:]}", file=sys.stderr)
        return None
    return rep_dir / "trial_1.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-model", required=True)
    ap.add_argument("--prior-file", default="models/pubchem20250312.prior.12.chkpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--zinc-holdout", default="data/zinc_holdout_low_pCMC_low_SurfTen.csv")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    args = ap.parse_args()

    zinc_holdout = pd.read_csv(args.zinc_holdout, usecols=["smiles_std"])
    zinc_set = set(zinc_holdout["smiles_std"])
    surfpro_holdout = pd.read_csv(args.surfpro_holdout, usecols=["SMILES_canonical"])
    surfpro_set = set(surfpro_holdout["SMILES_canonical"].apply(canon).dropna())

    print(f"ZINC holdout: {len(zinc_set)} molecules")
    print(f"SurfPro real holdout: {len(surfpro_set)} molecules\n")

    results = {}
    for config_name, params in CONFIGS.items():
        print(f"=== {config_name}: {params} ===")
        csvs = []
        for i in range(args.replicates):
            csv_path = run_replicate(config_name, params, i, args.tl_model, args.prior_file, args.steps, args.out_dir)
            if csv_path is not None:
                csvs.append(csv_path)
            print(f"  replicate {i}: {'ok' if csv_path else 'FAILED'}", flush=True)

        gen_all = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        gen_all["canon"] = gen_all["SMILES"].apply(canon)
        gen_unique = set(gen_all["canon"].dropna().unique())

        zinc_hits = gen_unique & zinc_set
        surfpro_hits = gen_unique & surfpro_set
        mean_score = gen_all["Score"].mean()

        results[config_name] = {
            "n_generated_total": len(gen_all),
            "n_unique_valid": len(gen_unique),
            "mean_score": mean_score,
            "zinc_hits": len(zinc_hits),
            "surfpro_hits": len(surfpro_hits),
            "surfpro_hit_smiles": sorted(surfpro_hits),
        }
        print(f"  unique valid: {len(gen_unique)}, mean Score: {mean_score:.4f}")
        print(f"  ZINC holdout hits: {len(zinc_hits)} / {len(zinc_set)}")
        print(f"  SurfPro real holdout hits: {len(surfpro_hits)} / {len(surfpro_set)}")
        print()

    print("=== SUMMARY ===")
    summary_df = pd.DataFrame(results).T
    print(summary_df[["n_unique_valid", "mean_score", "zinc_hits", "surfpro_hits"]].to_string())
    summary_df.to_csv(Path(args.out_dir) / "comparison_summary.csv")


if __name__ == "__main__":
    main()
