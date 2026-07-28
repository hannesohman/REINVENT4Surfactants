#!/usr/bin/env python3
"""
Single-combination driver for the 2026-07-27 production sweep:
2 (ZINC-plausibility on/off) x 4 (uncertainty mode: none/sm/lm/sm_lm) x
3 (Pareto: none/boost/gradient) = 24 combinations. For one combination:

1. Build that combo's scoring_functions + per-component weights (equal
   1/n_active across whichever of pCMC, SurfTen, pCMC_Uncertainty +
   SurfTen_Uncertainty, ZincPlausibility, Pareto/ParetoGradient are active in
   the geometric mean -- Loss Modulation's components are still declared with
   weight=0 so the LM patch can read their raw scores without them entering
   Score Modulation) and whether REINVENT_LM_ENABLED should be set.
2. Run an Optuna sweep (sigma/learning_rate/batch_size) maximizing mean
   top-100 `Score` -- never the rediscovery/holdout metrics (see README).
3. Run N production replicates at the best hyperparameters.
4. Evaluate with the full metric suite (validity, novelty, diversity,
   renormalized_score, nn_tanimoto_to_train, SurfPro/ZINC top-100
   rediscovery), per-replicate and pooled.

Storage: each RL run (HPO trial or production replicate) writes an ~85MB
model checkpoint. With 24 combos x 20 runs each that's 40+ GB if left in
place, so `checkpoints/`+`tb_logdir/` are deleted immediately after each run
(trial_1.csv/eval.json, which the resume-from-cache logic actually needs, are
kept).

Usage:
    python workflow/run_production_combo.py \
        --zinc on --unc-mode sm --pareto-mode none \
        --tl-model runs/validation/tl_only_.../generation_0/model/generation_0.model \
        --out-dir runs/production/zinc_on-unc_sm-pareto_none \
        --hpo-trials 15 --replicates 5 --steps 20
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd

sys.path.insert(0, "workflow")
from optuna_rl_search import make_toml, REINVENT_PYTHON
from run_replicated_eval import run_replicate, mean_std, SCALAR_METRICS
from evaluate_run import load_resources, evaluate

MODELS_PKL = "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl"
PCMC_PKG = {"pkl": "pcmc_model.joblib", "min_value": 0.0089955596692448, "max_value": 6.79588001734408}
SURFTEN_PKG = {"pkl": "final_model_surface_tension_avg.joblib", "min_value": 173.98984, "max_value": 594.85364}


def build_combo(zinc: str, unc_mode: str, pareto_mode: str):
    """Returns (scoring_functions, weights, lm_enabled)."""
    scoring_functions = {
        "pCMC": {"minimize": False, **PCMC_PKG},
        "SurfTen": {"minimize": True, **SURFTEN_PKG},
    }
    active = {"pCMC", "SurfTen"}

    if unc_mode in ("sm", "lm", "sm_lm"):
        # Declared for sm/lm/sm_lm alike -- LM (lm, sm_lm) reads these
        # components' raw scores regardless of whether SM (sm, sm_lm) also
        # includes them in the geometric mean.
        scoring_functions["pCMC_Uncertainty"] = {
            "minimize": True, "model_path": MODELS_PKL, "target": "pCMC",
            "min_value": 0.0476, "max_value": 0.6120,
        }
        scoring_functions["SurfTen_Uncertainty"] = {
            "minimize": True, "model_path": MODELS_PKL, "target": "surface_tension_avg",
            "min_value": 2.806, "max_value": 18.979,
        }
        if unc_mode in ("sm", "sm_lm"):
            active.add("pCMC_Uncertainty")
            active.add("SurfTen_Uncertainty")
        # unc_mode == "lm": declared above, weight forced to 0 below (not in `active`).

    if zinc == "on":
        scoring_functions["ZincPlausibility"] = {
            "minimize": False, "reference_path": "data/zinc_reference_profile.json.gz",
            "min_value": 0.0, "max_value": 1.0,
        }
        active.add("ZincPlausibility")

    if pareto_mode == "boost":
        scoring_functions["Pareto"] = {"SurfTen": SURFTEN_PKG, "pCMC": PCMC_PKG}
        active.add("Pareto")
    elif pareto_mode == "gradient":
        scoring_functions["ParetoGradient"] = {"SurfTen": SURFTEN_PKG, "pCMC": PCMC_PKG}
        active.add("ParetoGradient")

    n_active = len(active)
    weights = {name: (1.0 / n_active if name in active else 0.0) for name in scoring_functions}
    lm_enabled = unc_mode in ("lm", "sm_lm")

    return scoring_functions, weights, lm_enabled


def make_env(lm_enabled: bool, lm_components: str = "pCMC_Uncertainty,SurfTen_Uncertainty"):
    env = os.environ.copy()
    if lm_enabled:
        env["REINVENT_LM_ENABLED"] = "1"
        env["REINVENT_LM_COMPONENTS"] = lm_components
    else:
        env.pop("REINVENT_LM_ENABLED", None)
    return env


def cleanup_run_dir(run_dir: Path):
    """Delete the large, disposable-once-scored artifacts (model checkpoint +
    tensorboard logs), keeping the small ones the resume logic needs
    (trial_1.csv, eval.json, .toml, .log)."""
    for sub in ("checkpoints", "tb_logdir", "tb_logdir_0"):
        shutil.rmtree(run_dir / sub, ignore_errors=True)


def run_hpo_trial(trial: optuna.Trial, args, scoring_functions, weights, env, hpo_dir) -> float:
    sigma = trial.suggest_int("sigma", 50, 500, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    trial_dir = Path(hpo_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "checkpoints").mkdir(exist_ok=True)
    (trial_dir / "tb_logdir").mkdir(exist_ok=True)

    toml_text = make_toml(
        trial_dir, args.tl_model, args.prior_file, sigma, learning_rate,
        batch_size, args.steps, args.seed,
        scoring_functions=scoring_functions, weights=weights,
    )
    toml_path = trial_dir / "trial.toml"
    toml_path.write_text(toml_text)

    log_path = trial_dir / "trial.log"
    result = subprocess.run(
        [REINVENT_PYTHON, "-u", "workflow/reinvent_with_lm.py", "-l", str(log_path), str(toml_path)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"[hpo trial {trial.number}] FAILED\n{result.stderr[-4000:]}", file=sys.stderr, flush=True)
        cleanup_run_dir(trial_dir)
        raise optuna.TrialPruned()

    csv_path = trial_dir / "trial_1.csv"
    df = pd.read_csv(csv_path)
    top_k = df.sort_values("Score", ascending=False).head(100)
    value = float(top_k["Score"].mean())
    print(f"[hpo trial {trial.number}] sigma={sigma} lr={learning_rate:.2e} batch={batch_size} "
          f"-> mean_top100_score={value:.4f}", flush=True)

    cleanup_run_dir(trial_dir)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zinc", choices=["on", "off"], required=True)
    ap.add_argument("--unc-mode", choices=["none", "sm", "lm", "sm_lm"], required=True)
    ap.add_argument("--pareto-mode", choices=["none", "boost", "gradient"], required=True)
    ap.add_argument("--tl-model", required=True)
    ap.add_argument("--prior-file", default="models/pubchem20250312.prior.12.chkpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hpo-trials", type=int, default=15)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-csv", default="data/surfpro_expanded_trainval_only.csv")
    ap.add_argument("--train-smiles-col", default="SMILES_canonical")
    ap.add_argument("--surfpro-holdout", default="data/surfpro_real_holdout_test_split.csv")
    ap.add_argument("--zinc-reference", default="data/zinc_reference_profile.json.gz")
    ap.add_argument("--zinc-top100", default="data/zinc_top100_holdout.csv")
    ap.add_argument("--intdiv-sample", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_functions, weights, lm_enabled = build_combo(args.zinc, args.unc_mode, args.pareto_mode)
    env = make_env(lm_enabled)

    combo_meta = {
        "zinc": args.zinc, "unc_mode": args.unc_mode, "pareto_mode": args.pareto_mode,
        "scoring_functions": list(scoring_functions.keys()), "weights": weights, "lm_enabled": lm_enabled,
    }
    print(f"=== COMBO: {combo_meta} ===", flush=True)
    with open(out_dir / "combo_meta.json", "w") as f:
        json.dump(combo_meta, f, indent=2)

    # --- 1. HPO sweep (maximize mean top-100 Score; never rediscovery/holdout metrics) ---
    hpo_dir = out_dir / "hpo"
    hpo_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{hpo_dir}/optuna.db"
    study = optuna.create_study(study_name="combo_hpo", storage=storage, direction="maximize", load_if_exists=True)

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = args.hpo_trials - n_done
    if n_remaining > 0:
        print(f"running {n_remaining} more HPO trial(s) ({n_done} already complete)...", flush=True)
        study.optimize(
            lambda t: run_hpo_trial(t, args, scoring_functions, weights, env, hpo_dir),
            n_trials=n_remaining,
        )
    else:
        print(f"HPO already complete ({n_done}/{args.hpo_trials} trials).", flush=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No HPO trials completed successfully -- aborting this combo.", file=sys.stderr)
        sys.exit(1)
    best_params = study.best_params
    print(f"=== BEST HPO PARAMS: {best_params}  (value={study.best_value:.4f}) ===", flush=True)
    with open(out_dir / "best_hpo_params.json", "w") as f:
        json.dump({"best_params": best_params, "best_value": study.best_value}, f, indent=2)

    # --- 2. Production replicates at best hyperparameters ---
    print("loading fixed-cost eval resources...", flush=True)
    resources = load_resources(
        train_csv=args.train_csv, train_smiles_col=args.train_smiles_col,
        surfpro_holdout=args.surfpro_holdout, zinc_reference=args.zinc_reference,
        zinc_quintile_dir=None,
        surfpro_top100_csv=args.surfpro_holdout, zinc_top100_csv=args.zinc_top100,
    )

    per_rep_results = []
    csvs = []
    for i in range(args.replicates):
        rep_dir = out_dir / "production" / f"rep_{i}"
        print(f"--- production replicate {i} ---", flush=True)

        eval_path = rep_dir / "eval.json"
        csv_path = rep_dir / "trial_1.csv"
        if eval_path.exists() and csv_path.exists():
            print(f"  replicate {i}: cached", flush=True)
            gen_df = pd.read_csv(csv_path)
            with open(eval_path) as f:
                result = json.load(f)
        else:
            if not csv_path.exists():
                csv_path_result = run_replicate(
                    rep_dir, best_params, i, args.tl_model, args.prior_file, args.steps,
                    scoring_functions=scoring_functions, weights=weights, env=env,
                )
                if csv_path_result is None:
                    print(f"  replicate {i}: FAILED (generation)", flush=True)
                    continue
            gen_df = pd.read_csv(csv_path)
            result = evaluate(gen_df, "SMILES", resources, intdiv_sample=args.intdiv_sample)
            with open(eval_path, "w") as f:
                json.dump(result, f, indent=2)
            cleanup_run_dir(rep_dir)

        mean_score = float(gen_df["Score"].mean())
        result["replicate"] = i
        result["mean_score"] = mean_score
        per_rep_results.append(result)
        csvs.append(gen_df)
        print(
            f"  replicate {i}: ok  mean_score={mean_score:.4f}  validity={result['validity']:.3f}  "
            f"renormalized_score={result['renormalized_score']:.4f}  "
            f"surfpro_top100={result['surfpro_top100']['rate']:.3f}  "
            f"zinc_top100={result['zinc_top100']['rate']:.6f}",
            flush=True,
        )

    if not per_rep_results:
        print("ALL production replicates failed!", file=sys.stderr)
        sys.exit(1)

    agg = {m: mean_std([r[m] for r in per_rep_results]) for m in SCALAR_METRICS}
    agg["surfpro_top100"] = mean_std([r["surfpro_top100"]["rate"] for r in per_rep_results])
    agg["zinc_top100"] = mean_std([r["zinc_top100"]["rate"] for r in per_rep_results])

    pooled_df = pd.concat(csvs, ignore_index=True)
    pooled_result = evaluate(pooled_df, "SMILES", resources, intdiv_sample=args.intdiv_sample)
    pooled_result["mean_score"] = float(pooled_df["Score"].mean())

    final = {
        "combo": combo_meta, "best_hpo_params": best_params, "best_hpo_value": study.best_value,
        "per_replicate": per_rep_results, "aggregate": agg, "pooled": pooled_result,
    }
    with open(out_dir / "final_result.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n=== COMBO DONE: {combo_meta} ===", flush=True)
    for m in SCALAR_METRICS:
        print(f"  {m}: {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")
    print(f"  surfpro_top100: {agg['surfpro_top100']['mean']:.4f} +/- {agg['surfpro_top100']['std']:.4f}")
    print(f"  zinc_top100: {agg['zinc_top100']['mean']:.6f} +/- {agg['zinc_top100']['std']:.6f}")
    print(f"saved -> {out_dir / 'final_result.json'}")


if __name__ == "__main__":
    main()
