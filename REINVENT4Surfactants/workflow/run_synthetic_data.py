import subprocess
import time
import sys
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description="Run REINVENT multiples in parallel.")
parser.add_argument("--folder", action="store", dest="folder", default="baseline_runs", help="Path to the folder containing multiple subfolders")

args = parser.parse_args()

MAX_WORKERS = 7
run_first = 0  # Set to >0 to only run the first N multiples for testing

print(f"Starting run at {time.ctime()}", flush=True)
print(f"  Folder {args.folder}", flush=True)
print("", flush=True)


def run_single_multiple(folder):
    """Run a single REINVENT multiple. Returns (folder, success, elapsed, error_msg)."""
    csv_files = sorted(folder.glob("*.csv"))

    if csv_files:
        return (str(folder), None, 0, "skipped (CSV exists)")

    toml_files = sorted(folder.glob("*.toml"))
    if not toml_files:
        return (str(folder), False, 0, "no TOML file found")

    toml_file = toml_files[0]

    print(f"{'='*60}", flush=True)
    print(f"Running: {folder}", flush=True)
    print(f"{'='*60}", flush=True)


    # # Patch batch_size in the TOML config for better GPU utilization
    # toml_text = toml_file.read_text()
    # toml_text = re.sub(r'^batch_size\s*=\s*\d+', f'batch_size = {BATCH_SIZE}', toml_text, flags=re.MULTILINE)
    # toml_file.write_text(toml_text)

    log_path = folder / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = folder / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        subprocess.run(
            [sys.executable, "-u", "workflow/reinvent_with_lm.py", "-l", f"{log_path}/{folder.name}.log", str(toml_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        return (str(folder), True, elapsed, None)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        error_detail = f"exit code {e.returncode}"
        if e.stderr:
            error_detail += f"\n -- STDERR: -- \n{e.stderr.strip()[-10000:]}"
        if e.stdout:
            error_detail += f"\n -- STDOUT: -- \n{e.stdout.strip()[-10000:]}"
        return (str(folder), False, elapsed, error_detail)


# Run this file from the pubchem folder

# master_folder = Path("baseline_runs")
# multiple_folders = sorted(
#     f for f in master_folder.iterdir() 
#     if (f.is_dir() and f.name.startswith("multiple_"))
#     or (f.is_dir() and f.name.startswith("sigma_"))
# )
# print(f"Found {len(multiple_folders)} multiple folders in baseline_runs/", flush=True)


master_folder = Path(args.folder)
if master_folder.name.startswith("combo_"):
    subfolders = [master_folder]
else:
    subfolders = list(master_folder.glob("combo_*/"))

print(f"Found {len(subfolders)} subfolders in {master_folder}/", flush=True)

toml_folders = []
for folder in subfolders:
    print(f" - {folder}", flush=True)
    toml_folders.extend([
        p.parent for p in folder.rglob("*.toml")
        if any(part.startswith(("multiple_", "sigma_")) for part in p.parts)
    ])

print("\n")


# Filter to only folders that still need processing
to_run = [f for f in toml_folders if not sorted(f.glob("*.csv"))]


if run_first > 0:
    print(f"!!! LIMITED RUN [{run_first}] !!!")
    already_done = len(toml_folders) - len(to_run)
    to_run = to_run[:run_first]
    print(f"Already completed: {already_done}, To run: {len(to_run)} [LIMITED]", flush=True)
else:
    already_done = len(toml_folders) - len(to_run)
    print(f"Already completed: {already_done}, To run: {len(to_run)}", flush=True)


for folder in to_run:
    print(f" * {folder}", flush=True)
print(f"Running with {MAX_WORKERS} parallel workers\n", flush=True)




# Run in parallel
failed_runs = []
succeeded = 0
total_start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_single_multiple, folder): folder for folder in to_run}  # Limit to first 3 for testing

    for i, future in enumerate(as_completed(futures), 1):
        folder_path, success, elapsed, error_msg = future.result()
        folder_name = '/'.join(str(folder_path).split('/')[-3:])  # Show last three parts of the path for clarity

        if success is None:
            # Skipped
            print(f"[{i}/{len(to_run)}] {folder_name}: skipped (CSV exists)", flush=True)
        elif success:
            succeeded += 1
            print(f"[{i}/{len(to_run)}] {folder_name}: ✓ completed in {elapsed:.1f}s", flush=True)
        else:
            failed_runs.append((folder_path, error_msg))
            print(f"[{i}/{len(to_run)}] {folder_name}: ✗ failed after {elapsed:.1f}s \n — ERROR MESSAGE: — \n {error_msg}", flush=True)




total_elapsed = time.time() - total_start_time
print(f"\n{'='*60}", flush=True)
print(f"All runs completed in {total_elapsed/3600:.2f} hours", flush=True)
print(f"Success: {succeeded}/{len(to_run)}", flush=True)



failed_runs_log_path = master_folder / "failed_runs.log"
if failed_runs_log_path.exists():
    failed_runs_log_path.unlink()  # Remove old log if it exists

if failed_runs:
    with failed_runs_log_path.open("w") as log_file:
        for folder_path, error_msg in failed_runs:
            log_file.write(f"{folder_path}\n")

    print(f"\nFailed runs ({len(failed_runs)}):", flush=True)

    for folder_path, error_msg in failed_runs:
        print(f"  - {folder_path}", flush=True)

    print("")

    for folder_path, error_msg in failed_runs:
        print(f"  - {folder_path}: \n {error_msg}\n\n", flush=True)

sys.exit(0)
