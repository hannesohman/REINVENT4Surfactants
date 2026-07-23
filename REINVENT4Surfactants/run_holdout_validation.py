"""
Ground-truth validation run: identical to main.py's workflow, but the TL
training data excludes data/surfpro_expanded.csv's "test" split (130 real
surfactants, saved separately as data/surfpro_real_holdout_test_split.csv)
so we can check whether RL later (re)generates any of those real, ground-truth
surfactants it never saw during training.
"""
import json
from pathlib import Path
import pandas as pd
import time
import subprocess

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

config_parameters = json.load(open("config.json", "r"))

workflow_name = config_parameters["META"]["WORKFLOW_NAME"]
group_name = "validation"
run_name = "surfpro_holdout"
config_parameters["META"]["GROUP_NAME"] = group_name
config_parameters["META"]["RUN_NAME"] = run_name

run_id = f"{run_name}_{timestamp}"
config_parameters["META"]["RUN_ID"] = run_id

runs_output_folder = Path("runs")
group_folder = runs_output_folder / group_name
run_folder = group_folder / run_id

runs_output_folder.mkdir(exist_ok=True)
group_folder.mkdir(exist_ok=True)
run_folder.mkdir(exist_ok=True)

out_config_path = run_folder / "config.json"
json.dump(config_parameters, open(out_config_path, "w"), indent=4)

output_file_path = run_folder / f"{run_name}_{timestamp}.out"

jobscript_template_path = Path("templates/jobscript-run_template.sh")
jobscript_out_path = run_folder / f"jobscript-run_{run_name}_{timestamp}.sh"

with open(jobscript_template_path, "r") as f:
    jobscript = f.read()
    jobscript = jobscript.replace("{{CONFIG_PATH}}", str(out_config_path))
    jobscript = jobscript.replace("{{OUTPUT_PATH}}", str(output_file_path))

with open(jobscript_out_path, "w") as f:
    f.write(jobscript)

first_generation_data_folder = run_folder / "generation_0" / "data"
first_generation_data_folder.mkdir(exist_ok=True, parents=True)

surfpro_data = pd.read_csv("data/surfpro_expanded_trainval_only.csv")
surfpro_data.to_csv(first_generation_data_folder / "all_data.csv", index=False)

print(f"Run ID: {run_id}")
print(f"TL training data: data/surfpro_expanded_trainval_only.csv ({len(surfpro_data)} molecules, holdout test-split excluded)")

subprocess.run(["sbatch", str(jobscript_out_path)], check=True)
