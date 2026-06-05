import json
from pathlib import Path
import pandas as pd
import time
import subprocess
import sys

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# Load the config file
config_parameters = json.load(open("config.json", "r"))

# Create output folders
workflow_name = config_parameters["META"]["WORKFLOW_NAME"]
group_name = config_parameters["META"]["GROUP_NAME"]
run_name = config_parameters["META"]["RUN_NAME"]

run_id = f"{run_name}_{timestamp}"
config_parameters["META"]["RUN_ID"] = run_id

runs_output_folder = Path("runs")
group_folder = runs_output_folder / group_name
run_folder = group_folder / run_id

runs_output_folder.mkdir(exist_ok=True)
group_folder.mkdir(exist_ok=True)
run_folder.mkdir(exist_ok=True)

# Copy the config file to the output folder
out_config_path = run_folder / "config.json"
json.dump(config_parameters, open(out_config_path, "w"), indent=4)

# Define the output file path for the jobscript
output_file_path = run_folder / f"{run_name}_{timestamp}.out"

# Copy the jobscript template and fill in the paths
jobscript_template_path = Path("templates/jobscript-run_template.sh")
jobscript_out_path = run_folder / f"jobscript-run_{run_name}_{timestamp}.sh"

with open(jobscript_template_path, "r") as f:
    jobscript = f.read()
    jobscript = jobscript.replace("{{CONFIG_PATH}}", str(out_config_path))
    jobscript = jobscript.replace("{{OUTPUT_PATH}}", str(output_file_path))

with open(jobscript_out_path, "w") as f:
    f.write(jobscript)

# Create the first generation folder and copy the initial data
first_generation_data_folder = run_folder / "generation_0" / "data"
first_generation_data_folder.mkdir(exist_ok=True, parents=True)

surfpro_data = pd.read_csv("data/surfpro_expanded.csv")
surfpro_data.to_csv(first_generation_data_folder / "all_data.csv", index=False)


# Run the jobscript
subprocess.run(["sbatch", str(jobscript_out_path)], check=True)
