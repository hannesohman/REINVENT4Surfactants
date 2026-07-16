import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

import numpy as np
import pandas as pd

import time

from split_data import split_data
from generate_model_files import (
    generate_tl_toml,
    generate_tl_jobscript,

    generate_sl_jobscript_for_combo
)
from generate_combo_files import generate_combo_files
from prepare_next_generation import create_next_generation


def _format_elapsed(elapsed_seconds: float) -> str:
    return time.strftime('%Hh %Mmin %Ssec', time.gmtime(elapsed_seconds))


def _run_sbatch_wait(jobscript_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    start_time = time.time()
    cmd = ["sbatch", "--wait", str(jobscript_path)]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def submit_and_wait(
        meta_config: dict,
        generation: int,
        jobscript_path: Path
        ) -> None:
    start_time = time.time()

    run_name = meta_config.get("RUN_NAME")

    cmd = ["sbatch", "--wait", str(jobscript_path)]
    print(f"Submitting and waiting: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)

    if result.returncode != 0:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"[{run_name}] (GEN {generation}) ✗ failed after {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s) with output:", flush=True)
        print(f"STDOUT:\n{result.stdout.strip()}\n", flush=True)
        print("\n" + " -"*60 + "\n", flush=True)
        print(f"STDERR:\n{result.stderr.strip()}", flush=True)

        raise RuntimeError(
            f"Job failed for {jobscript_path} with exit code {result.returncode}."
        )

    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[{run_name}] (GEN {generation}) ✓ completed in {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s)")


def submit_many_and_wait(
        meta_config: dict,
        generation: int,
        jobscript_paths: list[Path]
        ) -> None:
    start_time = time.time()

    run_name = meta_config.get("RUN_NAME")
    run_id = meta_config.get("RUN_ID")

    failed_jobs: list[tuple[Path, int, str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, len(jobscript_paths))) as executor:
        futures = {}
        for jobscript_path in jobscript_paths:
            cmd = ["sbatch", "--wait", str(jobscript_path)]
            print(f"Submitting and waiting: {' '.join(cmd)}", flush=True)
            futures[executor.submit(_run_sbatch_wait, jobscript_path)] = jobscript_path

        for future in as_completed(futures):
            jobscript_path = futures[future]
            combo_name = jobscript_path.parent.name
            try:
                result, elapsed_time = future.result()
            except Exception as exc:
                failed_jobs.append((jobscript_path, -1, "", str(exc)))
                print(f"[{run_name}] (GEN {generation}) ✗ {combo_name} failed with exception: {exc}", flush=True)
                continue

            if result.returncode != 0:
                failed_jobs.append((jobscript_path, result.returncode, result.stdout.strip(), result.stderr.strip()))
                print(
                    f"[{run_name}] (GEN {generation}) ✗ {combo_name} failed in {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"[{run_name}] (GEN {generation}) ✓ {combo_name} completed in {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s)",
                    flush=True,
                )

    elapsed_time = time.time() - start_time
    if failed_jobs:
        print(f"[{run_name}] (GEN {generation}) ✗ {len(failed_jobs)} SL jobs failed after {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s)", flush=True)
        for failed_path, returncode, stdout, stderr in failed_jobs:
            print(f"Failed jobscript: {failed_path} (exit {returncode})", flush=True)
            print(f"STDOUT:\n{stdout}\n", flush=True)
            print("\n" + " -"*60 + "\n", flush=True)
            print(f"STDERR:\n{stderr}", flush=True)

        raise RuntimeError("One or more parallel SL jobs failed.")

    print(f"[{run_name}] (GEN {generation}) ✓ all SL jobs completed in {_format_elapsed(elapsed_time)} ({elapsed_time:.1f}s)", flush=True)

    
def check_and_remove_failed_runs(combo_folder: Path) -> bool:
    failed_runs_path = combo_folder / "failed_runs.log"
    if failed_runs_path.exists():
        need_rerun = True

        with open(failed_runs_path, "r") as f:
            failed_runs = f.readlines()

            if failed_runs:
                print(f"Found {len(failed_runs)} failed runs in {failed_runs_path}. Removing corresponding files and preparing for rerun.")

                for line in failed_runs:
                    folder_path = Path(line.strip())
                    folder_parts = folder_path.parts
                    multiple_name = folder_parts[-1]
                    file_path = folder_path / f"{multiple_name}_1.csv"
                    if file_path.exists():
                        print(f"Removing failed run file: {file_path}")
                        file_path.unlink()
                    else:
                        print(f"Warning: Failed run file not found for removal: {file_path}")
            else:
                need_rerun = False
                print(f"No failed runs found in {failed_runs_path}. No need to rerun.")
    else:
        need_rerun = False

    return need_rerun



def main(config) -> None:

    meta_config = config.get("META")
    parameter_config = config.get("PARAMETERS")
    weight_combos = config.get("WEIGHT_COMBOS")
    scoring_functions = config.get("SCORING_FUNCTIONS")
    
    try_start_time = time.time()
    print(f"\n" + f" [{time.strftime('%d/%m %H:%M:%S')}] STARTING REINVENT_4_SURFACTANTS [{meta_config.get("RUN_NAME")}] ".center(100, "="), flush=True)
    print(f"CONFIG:")
    print(f"  Group: {meta_config.get("GROUP_NAME")}")
    print(f"  Run Name: {meta_config.get("RUN_NAME")}    ({meta_config.get("RUN_ID")})")   
    print(f"  Start From Generation: {parameter_config.get("START_FROM")}")
    print("")
    print(f"  Epochs for TL: {parameter_config.get("N_EPOCHS") if parameter_config.get("N_EPOCHS") > 0 else "N/A (no TL)"}")
    print("")
    print(f"  Generations: {parameter_config.get("N_GENERATIONS")}")
    print(f"  Weight Combos: {len(weight_combos)}")
    print(f"  Multiples per Combo: {parameter_config.get("N_MULTIPLES_PER_GEN")}")
    print(f"  Steps per Multiple: {parameter_config.get("N_STEPS_PER_MULT")}")
    print(f"  Time per Combo: {parameter_config.get("COMBO_TIME")}")
    print(f"  Batch size: {parameter_config.get("BATCH_SIZE", 128)}")
    print(f"  = {parameter_config.get("N_GENERATIONS") * len(weight_combos) * parameter_config.get("N_MULTIPLES_PER_GEN") * parameter_config.get("N_STEPS_PER_MULT") * parameter_config.get("BATCH_SIZE", 128)} total oracle calls (ignoring TL epochs)")
    print("")
    print(f"  Sigma: {parameter_config.get("SIGMA")}")


    for generation in range(parameter_config.get("START_FROM"), parameter_config.get("N_GENERATIONS")):
        generation_start_time = time.time()
        

        print(f"\n" + f" [{time.strftime('%d/%m %H:%M:%S')}] Generation {generation} ".center(100, "="), flush=True)

        if parameter_config.get('DO_TL'):
            ## Step 1: Split data into training and validation sets 80/20
            split_data(meta_config, generation)



            ## Step 2: Generate and submit TL jobscript, wait for completion
            generate_tl_toml(meta_config, parameter_config, generation)
            tl_jobscript = generate_tl_jobscript(meta_config, generation)
            submit_and_wait(meta_config, generation, tl_jobscript)
            print("")

        else:
            print(f"[{meta_config.get('RUN_NAME')}] (GEN {generation}): Skipping TL phase as DO_TL is set to False in config.", flush=True)
            print(f"[{meta_config.get('RUN_NAME')}] (GEN {generation}): Using original prior instead.", flush=True)
            print("")

        ## Step 3: Generate combo files for SL multiples
        combo_folders = generate_combo_files(
            meta_config, 
            parameter_config, 
            generation, 
            scoring_functions, 
            weight_combos
            )



        ## Step 4: Submit one SL job per combo folder in parallel, and wait for all
        combo_folders_to_run = list(combo_folders)
        while combo_folders_to_run:
            sl_jobscripts = [
                generate_sl_jobscript_for_combo(meta_config, parameter_config, generation, combo_folder)
                for combo_folder in combo_folders_to_run
            ]
            submit_many_and_wait(meta_config, generation, sl_jobscripts)

            next_combo_folders_to_run = []
            for combo_folder in combo_folders_to_run:
                if check_and_remove_failed_runs(combo_folder):
                    next_combo_folders_to_run.append(combo_folder)

            combo_folders_to_run = next_combo_folders_to_run
            if len(combo_folders_to_run) > 0:
                print(f"[{meta_config.get('RUN_NAME')}] (GEN {generation}): {len(combo_folders_to_run)} combo folders need rerun due to failed runs. Retrying...", flush=True)
                print(f"- "*50 + f"\n", flush=True)

        print("")

        ## Step 5: Update next generation's all_data.csv with Pareto front from current generation
        create_next_generation(meta_config, generation)

        generation_end_time = time.time()
        generation_elapsed = generation_end_time - generation_start_time
        print(f"[Elapsed: {time.strftime('%Hh %Mmin %Ssec', time.gmtime(generation_elapsed))} ({generation_elapsed:.1f}s)]".center(100, "="), flush=True)



    try_end_time = time.time()
    total_elapsed = try_end_time - try_start_time
    total_days, total_remainder = divmod(total_elapsed, 86400)
    total_hours, total_remainder = divmod(total_remainder, 3600)
    total_minutes, total_seconds = divmod(total_remainder, 60)

    
    print(f"[{meta_config.get('RUN_NAME')}] Total elapsed time: {int(total_days)}d {int(total_hours)}h {int(total_minutes)}min {total_seconds:.1f}sec ({total_elapsed:.1f}s)".center(100, "="), flush=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides other args)")
    args = parser.parse_args()

    config_path = args.config
    if config_path is not None:
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Loading configuration from {config_path}", flush=True)
        
        with open(config_path, "r") as f:
            config = json.load(f)

        parameter_config = config.get("PARAMETERS")

        # overwrite combo time with a better estimate based on steps per multiple
        combo_time = f"{int(np.ceil(parameter_config.get('N_STEPS_PER_MULT')/(4*60)))+1}:00:00"
        config["PARAMETERS"]["COMBO_TIME"] = combo_time

        print(config, flush=True)
    else:
        raise FileNotFoundError(f"No config file found at: {config_path}")
    
    weight_combos = config.get("WEIGHT_COMBOS")
    weight_labels = [combo[0] for combo in weight_combos[0]]
    weight_labels.sort()

    scoring_functions = config.get("SCORING_FUNCTIONS")
    scoring_labels = list(scoring_functions.keys())
    scoring_labels.sort()

    if weight_labels != scoring_labels:
        raise ValueError(f"Weight combo labels {weight_labels} do not match scoring function labels {scoring_labels} in config.")

    main(config)



