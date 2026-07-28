#!/usr/bin/env python3
"""
Optuna sweep over REINVENT staged-learning (RL) hyperparameters -- sigma,
learning_rate, batch_size -- under the current uncertainty-aware objective
(pCMC, SurfTen, pCMC_Uncertainty, SurfTen_Uncertainty, equal weight).

Transfer learning is NOT repeated per trial: every trial starts from the same
already-fine-tuned TL checkpoint (--tl-model), so each trial only pays for the
RL steps themselves (a few minutes rather than TL's ~2-3 additional minutes).
All trials use the same seed, so differences in outcome reflect the
hyperparameters, not generation randomness.

Objective: mean `Score` (the geometric-mean composite REINVENT itself reports,
identical to config.json's WEIGHT_COMBOS objective) over the top-100 scoring
molecules generated in that trial. Maximize.

Usage (inside a job already holding one GPU):
    python workflow/optuna_rl_search.py \
        --tl-model runs/test/test_2026-07-17-10-34-54/generation_0/model/generation_0.model \
        --out-dir runs/optuna/rl_sweep_1 \
        --n-trials 30 --steps 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd

REINVENT_PYTHON = sys.executable

SCORING_FUNCTIONS = {
    # pCMC = -log10(CMC): HIGHER pCMC = LOWER CMC = more efficient surfactant,
    # so it's maximized (minimize=False) -- confirmed empirically 2026-07-21,
    # see README. (Was minimize=True until this fix -- a real bug.)
    "pCMC": {"minimize": False, "pkl": "pcmc_model.joblib",
              "min_value": 0.0089955596692448, "max_value": 6.79588001734408},
    "SurfTen": {"minimize": True, "pkl": "final_model_surface_tension_avg.joblib",
                 "min_value": 173.98984, "max_value": 594.85364},
    "pCMC_Uncertainty": {"minimize": True,
                          "model_path": "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl",
                          "target": "pCMC", "min_value": 0.0476, "max_value": 0.6120},
    "SurfTen_Uncertainty": {"minimize": True,
                             "model_path": "/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl",
                             "target": "surface_tension_avg", "min_value": 2.806, "max_value": 18.979},
    "ZincPlausibility": {"minimize": False,
                          "reference_path": "data/zinc_reference_profile.json.gz",
                          "min_value": 0.0, "max_value": 1.0},
}
WEIGHT = 0.2  # equal-weighted, matching config.json


def make_toml(trial_dir: Path, agent_file: str, prior_file: str, sigma: float,
              learning_rate: float, batch_size: int, steps: int, seed: int,
              scoring_functions: dict = None, weight: float = None,
              weights: dict = None) -> str:
    """`weight`: uniform weight applied to every component (backward-compatible
    default). `weights`: optional {name: weight} overrides for specific
    components (e.g. declaring pCMC_Uncertainty/SurfTen_Uncertainty with
    weight=0 for Loss-Modulation-only runs, while other terms use `weight`)."""
    if scoring_functions is None:
        scoring_functions = SCORING_FUNCTIONS
    if weight is None:
        weight = WEIGHT
    if weights is None:
        weights = {}

    components = ""
    for name, pkg in scoring_functions.items():
        this_weight = weights.get(name, weight)
        if name.endswith("_Uncertainty"):
            components += f"""
[[stage.scoring.component]]
[stage.scoring.component.UncertaintyPenalty]
[[stage.scoring.component.UncertaintyPenalty.endpoint]]
name = "{name}"
weight = {this_weight:.2f}
params.model_path = "{pkg['model_path']}"
params.target = "{pkg['target']}"
params.min_value = {pkg['min_value']}
params.max_value = {pkg['max_value']}
params.minimize = {str(pkg['minimize']).lower()}
"""
        elif name == "ZincPlausibility":
            components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ZincPlausibility]
[[stage.scoring.component.ZincPlausibility.endpoint]]
name = "{name}"
weight = {this_weight:.2f}
params.reference_path = "{pkg['reference_path']}"
params.min_value = {pkg['min_value']}
params.max_value = {pkg['max_value']}
params.minimize = {str(pkg['minimize']).lower()}
"""
        elif name == "Pareto":
            # Self-referential data_path: points at THIS trial's own
            # progressively-written summary CSV (same pattern as
            # generate_combo_files.py's Pareto branch), so the Pareto front
            # evolves from this trial's own earlier steps.
            components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ParetoBoost]
[[stage.scoring.component.ParetoBoost.endpoint]]
name = "{name}"
weight = {this_weight:.2f}
params.data_path = "{trial_dir}/trial_1.csv"
params.skip = {pkg.get("skip", 0)}
params.SurfTen_model_path = "models/{pkg['SurfTen']['pkl']}"
params.SurfTen_min_value = {pkg['SurfTen']['min_value']}
params.SurfTen_max_value = {pkg['SurfTen']['max_value']}
params.pCMC_model_path = "models/{pkg['pCMC']['pkl']}"
params.pCMC_min_value = {pkg['pCMC']['min_value']}
params.pCMC_max_value = {pkg['pCMC']['max_value']}
params.base_score = {pkg.get("base_score", 1.0)}
params.boost_factor = {pkg.get("boost_factor", 1.0)}
"""
        elif name == "ParetoGradient":
            components += f"""
[[stage.scoring.component]]
[stage.scoring.component.ParetoGradient]
[[stage.scoring.component.ParetoGradient.endpoint]]
name = "{name}"
weight = {this_weight:.2f}
params.data_path = "{trial_dir}/trial_1.csv"
params.distance_exponent = {pkg.get("distance_exponent", 1.0)}
params.max_score = {pkg.get("max_score", 1.0)}
params.SurfTen_model_path = "models/{pkg['SurfTen']['pkl']}"
params.SurfTen_min_value = {pkg['SurfTen']['min_value']}
params.SurfTen_max_value = {pkg['SurfTen']['max_value']}
params.pCMC_model_path = "models/{pkg['pCMC']['pkl']}"
params.pCMC_min_value = {pkg['pCMC']['min_value']}
params.pCMC_max_value = {pkg['pCMC']['max_value']}
"""
        else:
            components += f"""
[[stage.scoring.component]]
[stage.scoring.component.SurrogateModel]
[[stage.scoring.component.SurrogateModel.endpoint]]
name = "{name}"
weight = {this_weight:.2f}
params.model_path = "models/{pkg['pkl']}"
params.min_value = {pkg['min_value']}
params.max_value = {pkg['max_value']}
params.minimize = {str(pkg['minimize']).lower()}
"""

    return f"""
run_type = "staged_learning"
device = "cuda:0"
seed = {seed}
tb_logdir = "{trial_dir}/tb_logdir"
json_out_config = "{trial_dir}/out_config.json"

[parameters]
prior_file = "{prior_file}"
agent_file = "{agent_file}"
summary_csv_prefix = "{trial_dir}/trial"
batch_size = {batch_size}
use_checkpoint = false

[learning_strategy]
type = "dap"
sigma = {sigma}
rate = {learning_rate}

[[stage]]
max_score = 1
max_steps = {steps}
chkpt_file = "{trial_dir}/checkpoints/trial.chkpt"

[stage.scoring]
type = "geometric_mean"
{components}
"""


def run_trial(trial: optuna.Trial, args) -> float:
    sigma = trial.suggest_int("sigma", 50, 500, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    trial_dir = Path(args.out_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "checkpoints").mkdir(exist_ok=True)
    (trial_dir / "tb_logdir").mkdir(exist_ok=True)

    toml_text = make_toml(
        trial_dir, args.tl_model, args.prior_file, sigma, learning_rate,
        batch_size, args.steps, args.seed,
    )
    toml_path = trial_dir / "trial.toml"
    toml_path.write_text(toml_text)

    log_path = trial_dir / "trial.log"
    result = subprocess.run(
        [REINVENT_PYTHON, "-u", "workflow/reinvent_with_lm.py", "-l", str(log_path), str(toml_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[trial {trial.number}] FAILED\n{result.stderr[-4000:]}", file=sys.stderr)
        raise optuna.TrialPruned()

    csv_path = trial_dir / "trial_1.csv"
    df = pd.read_csv(csv_path)
    top_k = df.sort_values("Score", ascending=False).head(100)
    value = float(top_k["Score"].mean())

    print(f"[trial {trial.number}] sigma={sigma:.1f} lr={learning_rate:.2e} "
          f"batch={batch_size} -> mean_top100_score={value:.4f}", flush=True)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl-model", required=True, help="already fine-tuned TL checkpoint, reused for every trial")
    ap.add_argument("--prior-file", default="models/pubchem20250312.prior.12.chkpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--study-name", default="rl_sweep")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.out_dir}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize", load_if_exists=True,
    )
    study.optimize(lambda t: run_trial(t, args), n_trials=args.n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print("\n=== BEST TRIAL ===")
        print(f"value: {study.best_value:.4f}")
        print(f"params: {study.best_params}")
    else:
        print("\nNo trials completed successfully -- check trial logs under", args.out_dir)

    trials_df = study.trials_dataframe()
    trials_csv = Path(args.out_dir) / "all_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nall trials -> {trials_csv}")


if __name__ == "__main__":
    main()
