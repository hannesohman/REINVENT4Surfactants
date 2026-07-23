"""
Run transfer learning only (no RL), on the holdout-respecting dataset
(data/surfpro_expanded_trainval_only.csv -- excludes the 130 real surfactants
in data/surfpro_real_holdout_test_split.csv), and wait for it to finish.

Produces a single TL checkpoint meant to be reused across multiple downstream
RL configurations (Optuna trials, final comparison runs) without repeating the
~2-3 minute TL step each time.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "workflow")
from generate_model_files import generate_tl_toml, generate_tl_jobscript
from split_data import split_data
from run_try import submit_and_wait

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

config_parameters = json.load(open("config.json", "r"))
parameter_config = config_parameters["PARAMETERS"]

meta_config = {
    "WORKFLOW_NAME": "workflow",
    "GROUP_NAME": "validation",
    "RUN_NAME": "tl_only",
}
run_id = f"tl_only_{timestamp}"
meta_config["RUN_ID"] = run_id

run_folder = Path("runs") / meta_config["GROUP_NAME"] / run_id
data_folder = run_folder / "generation_0" / "data"
data_folder.mkdir(parents=True, exist_ok=True)

surfpro_data = pd.read_csv("data/surfpro_expanded_trainval_only.csv")
surfpro_data.to_csv(data_folder / "all_data.csv", index=False)
print(f"TL training data: data/surfpro_expanded_trainval_only.csv ({len(surfpro_data)} molecules)")

split_data(meta_config, 0)
tl_toml_path = generate_tl_toml(meta_config, parameter_config, 0)
tl_jobscript = generate_tl_jobscript(meta_config, 0)

print(f"Submitting TL job: {tl_jobscript}")
submit_and_wait(meta_config, 0, tl_jobscript)

model_path = run_folder / "generation_0" / "model" / "generation_0.model"
print(f"\nTL CHECKPOINT: {model_path}")
