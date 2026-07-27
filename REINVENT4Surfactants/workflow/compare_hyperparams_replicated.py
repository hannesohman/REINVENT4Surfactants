#!/usr/bin/env python3
"""
Head-to-head "default" vs "optimized" hyperparameter comparison, 5 replicates
each (matching run_replicated_eval.py's per-replicate methodology, extended
to two configs), under the CURRENT 5-term objective (pCMC, SurfTen,
pCMC_Uncertainty, SurfTen_Uncertainty, ZincPlausibility -- see
optuna_rl_search.SCORING_FUNCTIONS / config.json). Optimizes for maximum mean
`Score` only -- rediscovery/holdout metrics are read out afterward, never
targeted.

"default": config.json's current SIGMA/LEARNING_RATE/BATCH_SIZE.
"optimized": the Optuna sweep's best trial (sigma=225, lr=9.79e-05,
batch=512) -- found under the earlier 4-term objective (pre-ZincPlausibility);
reused here as-is rather than re-sweeping, per user request for "a simple
optimisation run".

Fixed-cost evaluation resources (training profile, ZINC tiers, ZINC
reference) are loaded ONCE and reused across all 10 replicate evaluations.

Usage:
    python workflow/compare_hyperparams_replicated.py \
        --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
        --out-dir runs/compare_replicated_1 --replicates 5 --steps 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "workflow")
from run_replicated_eval import run_replicate, mean_std, SCALAR_METRICS
from evaluate_run import load_resources, evaluate

CONFIGS = {
    "default": {"sigma": 120, "learning_rate": 0.00038716608106033025, "batch_size": 256},
    "optimized": {"sigma": 225, "learning_rate": 9.788453643515026e-05, "batch_size": 512},
}

# Two objective variants to isolate the effect of ZincPlausibility itself:
#  - "with_zinc": current config.json default -- pCMC/SurfTen (each with UWO
#    uncertainty baked in, see below) + ZincPlausibility, 1/3 weight each.
#  - "no_zinc": pCMC/SurfTen only (UWO uncertainty still applied per-property),
#    0.5 weight each -- isolates ZincPlausibility's effect from uncertainty
#    handling, which is otherwise identical between the two variants.
#
# Uncertainty is combined via UWO (Coste et al. 2024, ICLR): combined =
# point_score - lambda_weight * uncertainty_score, inside a single
# UncertaintyWeightedScore component per property -- replaces the old separate
# pCMC_Uncertainty/SurfTen_Uncertainty geometric-mean terms (2026-07-22).
_MODELS_PKL = "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl"

SCORING_FUNCTIONS_WITH_ZINC = {
    "pCMC": {"minimize": False, "pkl": "pcmc_model.joblib",  # HIGHER pCMC = LOWER CMC = better; see README
              "min_value": 0.0089955596692448, "max_value": 6.79588001734408,
              "uncertainty_model_path": _MODELS_PKL, "uncertainty_target": "pCMC",
              "uncertainty_min_value": 0.0476, "uncertainty_max_value": 0.6120,
              "lambda_weight": 0.5},
    "SurfTen": {"minimize": True, "pkl": "final_model_surface_tension_avg.joblib",
                 "min_value": 173.98984, "max_value": 594.85364,
                 "uncertainty_model_path": _MODELS_PKL, "uncertainty_target": "surface_tension_avg",
                 "uncertainty_min_value": 2.806, "uncertainty_max_value": 18.979,
                 "lambda_weight": 0.5},
    "ZincPlausibility": {"minimize": False,
                          "reference_path": "data/zinc_reference_profile.json.gz",
                          "min_value": 0.0, "max_value": 1.0},
}
WEIGHT_WITH_ZINC = 1 / 3

SCORING_FUNCTIONS_NO_ZINC = {
    k: v for k, v in SCORING_FUNCTIONS_WITH_ZINC.items() if k != "ZincPlausibility"
}
WEIGHT_NO_ZINC = 0.5

VARIANTS = {
    "with_zinc": (SCORING_FUNCTIONS_WITH_ZINC, WEIGHT_WITH_ZINC),
    "no_zinc": (SCORING_FUNCTIONS_NO_ZINC, WEIGHT_NO_ZINC),
}


def run_config(config_name, params, args, resources, scoring_functions, weight):
    print(f"\n=== CONFIG: {config_name}  {params}  variant={args.variant} ===", flush=True)
    per_rep_results = []
    csvs = []
    for i in range(args.replicates):
        rep_dir = Path(args.out_dir) / config_name / f"rep_{i}"
        print(f"--- {config_name} replicate {i} ---", flush=True)

        eval_path = rep_dir / "eval.json"
        csv_path = rep_dir / "trial_1.csv"
        if eval_path.exists() and csv_path.exists():
            print(f"  replicate {i}: cached, reusing existing eval.json/trial_1.csv", flush=True)
            gen_df = pd.read_csv(csv_path)
            with open(eval_path) as f:
                result = json.load(f)
            # JSON round-tripping turns int tier keys into strings -- normalize
            # back so cached and freshly-evaluated replicates are consistent.
            result["surfpro_tier_hits"] = {int(k): v for k, v in result["surfpro_tier_hits"].items()}
            result["zinc_tier_hits"] = {int(k): v for k, v in result["zinc_tier_hits"].items()}
        else:
            if not csv_path.exists():
                csv_path_result = run_replicate(
                    rep_dir, params, i, args.tl_model, args.prior_file, args.steps,
                    scoring_functions=scoring_functions, weight=weight,
                )
                if csv_path_result is None:
                    print(f"  replicate {i}: FAILED (generation)", flush=True)
                    continue
            else:
                print(f"  replicate {i}: reusing existing trial_1.csv, re-evaluating", flush=True)

            gen_df = pd.read_csv(csv_path)
            result = evaluate(gen_df, "SMILES", resources, intdiv_sample=args.intdiv_sample)
            with open(eval_path, "w") as f:
                json.dump(result, f, indent=2)

        mean_score = float(gen_df["Score"].mean())
        csvs.append(gen_df)

        result["replicate"] = i
        result["mean_score"] = mean_score
        per_rep_results.append(result)
        print(
            f"  replicate {i}: ok  mean_score={mean_score:.4f}  validity={result['validity']:.3f}  "
            f"surfpro_top2={result['surfpro_top2_vs_bottom2']['top2']['rate']:.3f}  "
            f"surfpro_bottom2={result['surfpro_top2_vs_bottom2']['bottom2']['rate']:.3f}",
            flush=True,
        )

    if not per_rep_results:
        print(f"  ALL replicates failed for {config_name}!", file=sys.stderr)
        return None, None

    agg = {m: mean_std([r[m] for r in per_rep_results]) for m in SCALAR_METRICS}
    tiers = sorted(per_rep_results[0]["surfpro_tier_hits"].keys(), key=int)
    agg["surfpro_tier_hits"] = {
        t: mean_std([r["surfpro_tier_hits"][t]["rate"] for r in per_rep_results]) for t in tiers
    }
    agg["surfpro_top2_vs_bottom2"] = {
        "top2": mean_std([r["surfpro_top2_vs_bottom2"]["top2"]["rate"] for r in per_rep_results]),
        "bottom2": mean_std([r["surfpro_top2_vs_bottom2"]["bottom2"]["rate"] for r in per_rep_results]),
    }
    zinc_tiers = sorted(per_rep_results[0]["zinc_tier_hits"].keys(), key=int)
    agg["zinc_tier_hits"] = {
        t: mean_std([r["zinc_tier_hits"][t]["rate"] for r in per_rep_results]) for t in zinc_tiers
    }

    # Pooled (all replicates' unique molecules combined) result, for
    # comparability with earlier pooled-3-replicate Findings tables.
    pooled_df = pd.concat(csvs, ignore_index=True)
    pooled_result = evaluate(pooled_df, "SMILES", resources, intdiv_sample=args.intdiv_sample)
    pooled_result["mean_score"] = float(pooled_df["Score"].mean())

    return {"per_replicate": per_rep_results, "aggregate": agg}, pooled_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-model", required=True)
    ap.add_argument("--prior-file", default="models/pubchem20250312.prior.12.chkpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--train-smiles-col", default="SMILES_canonical")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--zinc-quintile-dir", default="data")
    ap.add_argument("--zinc-reference", default="data/zinc_reference_profile.json.gz")
    ap.add_argument("--intdiv-sample", type=int, default=2000)
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), default="with_zinc",
                     help="with_zinc: current 5-term objective. no_zinc: original 4-term objective, ZincPlausibility excluded.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_functions, weight = VARIANTS[args.variant]

    print("loading fixed-cost resources once (train profile, ZINC tiers, ZINC reference)...", flush=True)
    resources = load_resources(
        args.train_csv, args.train_smiles_col, args.surfpro_holdout,
        args.zinc_quintile_dir, args.zinc_reference,
    )

    all_results = {}
    for config_name, params in CONFIGS.items():
        replicated, pooled = run_config(config_name, params, args, resources, scoring_functions, weight)
        if replicated is None:
            continue
        all_results[config_name] = {"params": params, "replicated": replicated, "pooled": pooled}
        with open(out_dir / f"{config_name}_full.json", "w") as f:
            json.dump(all_results[config_name], f, indent=2)

    print("\n\n=== HEAD-TO-HEAD SUMMARY (mean +/- std across replicates) ===")
    for config_name, res in all_results.items():
        agg = res["replicated"]["aggregate"]
        print(f"\n[{config_name}]  params={res['params']}")
        for m in SCALAR_METRICS:
            print(f"  {m}: {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")
        print(f"  surfpro top2: {agg['surfpro_top2_vs_bottom2']['top2']['mean']:.4f} +/- {agg['surfpro_top2_vs_bottom2']['top2']['std']:.4f}")
        print(f"  surfpro bottom2: {agg['surfpro_top2_vs_bottom2']['bottom2']['mean']:.4f} +/- {agg['surfpro_top2_vs_bottom2']['bottom2']['std']:.4f}")
        print(f"  pooled surfpro top2 rate: {res['pooled']['surfpro_top2_vs_bottom2']['top2']['rate']:.4f}")
        print(f"  pooled surfpro bottom2 rate: {res['pooled']['surfpro_top2_vs_bottom2']['bottom2']['rate']:.4f}")
        print(f"  pooled n_unique_valid: {res['pooled']['n_unique_valid']}")

    with open(out_dir / "comparison_full.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved -> {out_dir / 'comparison_full.json'}")


if __name__ == "__main__":
    main()
