#!/bin/env bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius-cpu     # orchestrator only submits/waits on other jobs, no GPU work here
#SBATCH -t 40:00:00
#SBATCH -J test_main
#SBATCH -o {{OUTPUT_PATH}}

PY=/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python

"$PY" -u workflow/run_try.py \
 --config {{CONFIG_PATH}}
