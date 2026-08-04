#!/bin/bash
# Generates and submits one sbatch job per combination of the production
# sweep: ZINC-similarity off only (2026-08-03: dropped as a varied dimension,
# not just excluded from plots) x 2 (uncertainty mode: none/lm -- SM/SM+LM
# dropped 2026-08-03 as not effective) x 3 (Pareto) = 6 jobs. Each job is
# self-contained (its own Optuna HPO sweep + production replicates +
# evaluation), so all 6 can run in parallel as SLURM schedules GPUs.
#
# batch_size is searched over {10,50,100,200,500}; step count is derived per
# trial/replicate as a fixed oracle budget (10000 proposed molecules) divided
# by that run's batch_size, so every run proposes the same number of
# molecules regardless of the batch/step split (2026-08-03).
#
# Usage: bash workflow/submit_all_combos.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TL_MODEL="runs/validation/tl_only_2026-07-29-12-30-48/generation_0/model/generation_0.model"
JOBS_DIR="runs/production_jobs"
OUT_BASE="runs/production"
HPO_TRIALS=15
REPLICATES=20
ORACLE_BUDGET=10000
WALLTIME="24:00:00"

mkdir -p "$JOBS_DIR" "$OUT_BASE"

for zinc in off; do
  for unc in none lm; do
    for pareto in none boost gradient; do
      name="zinc_${zinc}-unc_${unc}-pareto_${pareto}"
      script="$JOBS_DIR/${name}.sh"
      out_dir="$OUT_BASE/$name"

      cat > "$script" << EOF
#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -t ${WALLTIME}
#SBATCH -J prod_${name}
#SBATCH -o ${JOBS_DIR}/${name}_%j.out

export PYTHONPATH=/home/x_ribec/pylibs
PY=/proj/berzelius-2026-62/users/x_ribec/software/reinvent4-env/bin/python

"\$PY" -u gpu_keepalive.py --mat_size 8192 --sleep_time 0 &
KEEPALIVE_PID=\$!

"\$PY" -u workflow/run_production_combo.py \\
    --zinc ${zinc} --unc-mode ${unc} --pareto-mode ${pareto} \\
    --tl-model ${TL_MODEL} \\
    --out-dir ${out_dir} \\
    --hpo-trials ${HPO_TRIALS} --replicates ${REPLICATES} --oracle-budget ${ORACLE_BUDGET}

kill \$KEEPALIVE_PID
EOF

      jobid=$(sbatch --parsable "$script")
      echo "$name -> job $jobid"
    done
  done
done
