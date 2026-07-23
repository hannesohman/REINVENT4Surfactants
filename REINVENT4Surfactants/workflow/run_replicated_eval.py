#!/usr/bin/env python3
"""
Run N independent RL replicates from a fixed TL checkpoint, using
config.json's current default hyperparameters and scoring functions (as of
2026-07-21: the 5-term objective including ZincPlausibility, via
optuna_rl_search.make_toml/SCORING_FUNCTIONS), then run the full
evaluate_run.py metric suite on EACH replicate SEPARATELY (not pooled) so we
can report mean +/- std across replicates -- error bars -- for every metric.

Usage:
    python workflow/run_replicated_eval.py \
        --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
        --out-dir runs/replicated_eval_1 --replicates 5 --steps 20
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "workflow")
from optuna_rl_search import make_toml, REINVENT_PYTHON
from evaluate_run import load_resources, evaluate

DEFAULT_PARAMS = {"sigma": 120, "learning_rate": 0.00038716608106033025, "batch_size": 256}

SCALAR_METRICS = [
    "validity", "uniqueness", "novelty", "internal_diversity",
    "frag_similarity_train", "scaf_similarity_train",
    "frag_similarity_zinc", "scaf_similarity_zinc",
    "mean_score",
]


def run_replicate(rep_dir: Path, params, replicate_idx, tl_model, prior_file, steps,
                   scoring_functions=None, weight=None):
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "checkpoints").mkdir(exist_ok=True)
    (rep_dir / "tb_logdir").mkdir(exist_ok=True)

    toml_text = make_toml(
        rep_dir, tl_model, prior_file, params["sigma"], params["learning_rate"],
        params["batch_size"], steps, seed=replicate_idx,
        scoring_functions=scoring_functions, weight=weight,
    )
    toml_path = rep_dir / "rep.toml"
    toml_path.write_text(toml_text)

    log_path = rep_dir / "rep.log"
    result = subprocess.run(
        [REINVENT_PYTHON, "-u", "-m", "reinvent", "-l", str(log_path), str(toml_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[rep {replicate_idx}] FAILED\n{result.stderr[-4000:]}", file=sys.stderr)
        return None
    return rep_dir / "trial_1.csv"


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-model", required=True)
    ap.add_argument("--prior-file", default="models/pubchem20250312.prior.12.chkpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=DEFAULT_PARAMS["sigma"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_PARAMS["batch_size"])
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--train-smiles-col", default="SMILES_canonical")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--zinc-quintile-dir", default="data")
    ap.add_argument("--zinc-reference", default="data/zinc_reference_profile.json.gz")
    ap.add_argument("--intdiv-sample", type=int, default=2000)
    args = ap.parse_args()

    params = {"sigma": args.sigma, "learning_rate": args.learning_rate, "batch_size": args.batch_size}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading fixed-cost resources once (train profile, ZINC tiers, ZINC reference)...", flush=True)
    resources = load_resources(
        args.train_csv, args.train_smiles_col, args.surfpro_holdout,
        args.zinc_quintile_dir, args.zinc_reference,
    )

    per_rep_results = []
    for i in range(args.replicates):
        rep_dir = out_dir / f"rep_{i}"
        print(f"=== replicate {i} ===", flush=True)

        eval_path = rep_dir / "eval.json"
        csv_path = rep_dir / "trial_1.csv"
        if eval_path.exists() and csv_path.exists():
            print(f"  replicate {i}: cached, reusing existing eval.json/trial_1.csv", flush=True)
            gen_df = pd.read_csv(csv_path)
            with open(eval_path) as f:
                result = json.load(f)
            result["surfpro_tier_hits"] = {int(k): v for k, v in result["surfpro_tier_hits"].items()}
            result["zinc_tier_hits"] = {int(k): v for k, v in result["zinc_tier_hits"].items()}
        else:
            if not csv_path.exists():
                csv_path_result = run_replicate(rep_dir, params, i, args.tl_model, args.prior_file, args.steps)
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
        print("No replicates succeeded.", file=sys.stderr)
        sys.exit(1)

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

    print(f"\n=== AGGREGATE ({len(per_rep_results)} replicates, mean +/- std) ===")
    for m in SCALAR_METRICS:
        print(f"{m}: {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")
    print("surfpro tier hit rates:")
    for t in tiers:
        print(f"  tier {t}: {agg['surfpro_tier_hits'][t]['mean']:.4f} +/- {agg['surfpro_tier_hits'][t]['std']:.4f}")
    print(f"surfpro top2: {agg['surfpro_top2_vs_bottom2']['top2']['mean']:.4f} +/- {agg['surfpro_top2_vs_bottom2']['top2']['std']:.4f}")
    print(f"surfpro bottom2: {agg['surfpro_top2_vs_bottom2']['bottom2']['mean']:.4f} +/- {agg['surfpro_top2_vs_bottom2']['bottom2']['std']:.4f}")
    print("zinc tier hit rates:")
    for t in zinc_tiers:
        print(f"  tier {t}: {agg['zinc_tier_hits'][t]['mean']:.6f} +/- {agg['zinc_tier_hits'][t]['std']:.6f}")

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump({"per_replicate": per_rep_results, "aggregate": agg}, f, indent=2)
    print(f"\nsaved -> {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
