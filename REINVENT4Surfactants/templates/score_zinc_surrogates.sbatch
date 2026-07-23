#!/bin/env bash
#SBATCH -A berzelius-2026-62
#SBATCH -p berzelius-cpu
#SBATCH -t 6:00:00
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -J zinc_surrogate_score
#SBATCH -o /proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants/data/ZINC/score_zinc_%j.out
#SBATCH -e /proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants/data/ZINC/score_zinc_%j.err

PY=/proj/berzelius-2026-62/users/x_ribec/software/surf-surrogate-env/bin/python
REPO=/proj/berzelius-2026-62/users/x_ribec/surfactant_generation/REINVENT4Surfactants

"$PY" -u "$REPO/workflow/score_zinc_surrogates.py" \
  --input "$REPO/data/ZINC/zinc_instock_combined.csv.gz" \
  --output-dir "$REPO/data/ZINC/scored" \
  --models-path /proj/berzelius-2026-62/users/x_ribec/surfactant-surrogates/SurfPro-MD/surrogate-models/models.pkl \
  --chunk-size 500000 \
  --workers 64

"$PY" -u "$REPO/workflow/combine_zinc_chunks.py" \
  --chunks-dir "$REPO/data/ZINC/scored/chunks" \
  --out "$REPO/data/ZINC/zinc_scored_9props.csv.gz"
