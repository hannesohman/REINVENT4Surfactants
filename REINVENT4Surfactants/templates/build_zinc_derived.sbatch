#!/bin/env bash
#SBATCH -A berzelius-2026-62
#SBATCH -p berzelius-cpu
#SBATCH -t 2:00:00
#SBATCH -c 48
#SBATCH --mem=150G
#SBATCH -J zinc_build_derived
#SBATCH -o /proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants/data/ZINC/build_derived_%j.out
#SBATCH -e /proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants/data/ZINC/build_derived_%j.err

# Regenerates the ZINC-derived holdout/quintile files from
# data/ZINC/zinc_scored_9props.csv.gz -- run after score_zinc_surrogates.sbatch
# (see that job's completion). Submit with:
#   sbatch --dependency=afterok:<score_zinc_job_id> templates/build_zinc_derived.sbatch

PY=/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python
REPO=/proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants
TRAIN_CSV=/proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/SurfPro-MD.csv

"$PY" -u "$REPO/workflow/build_zinc_holdouts.py" \
  --scored "$REPO/data/ZINC/zinc_scored_9props.csv.gz" \
  --train-csv "$TRAIN_CSV" \
  --out-dir "$REPO/data" \
  --top-n 400

"$PY" -u "$REPO/workflow/build_zinc_reference.py" \
  --scored "$REPO/data/ZINC/zinc_scored_9props.csv.gz" \
  --train-csv "$TRAIN_CSV" \
  --out-dir "$REPO/data" \
  --reference-n 200000
