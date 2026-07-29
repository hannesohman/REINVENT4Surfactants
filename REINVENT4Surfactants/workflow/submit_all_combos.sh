#!/bin/bash
# Generates and submits one sbatch job per combination of the 2026-07-27
# production sweep: 2 (ZINC) x 4 (uncertainty mode) x 3 (Pareto) = 24 jobs.
# Each job is self-contained (its own Optuna HPO sweep + production replicates
# + evaluation), so all 24 can run in parallel as SLURM schedules GPUs.
#
# Usage: bash workflow/submit_all_combos.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TL_MODEL="runs/validation/tl_only_2026-07-29-12-30-48/generation_0/model/generation_0.model"
JOBS_DIR="runs/production_jobs"
OUT_BASE="runs/production"
HPO_TRIALS=15
REPLICATES=5
STEPS=20
WALLTIME="08:00:00"

mkdir -p "$JOBS_DIR" "$OUT_BASE"

for zinc in on off; do
  for unc in none sm lm sm_lm; do
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
    --hpo-trials ${HPO_TRIALS} --replicates ${REPLICATES} --steps ${STEPS}

kill \$KEEPALIVE_PID
EOF

      jobid=$(sbatch --parsable "$script")
      echo "$name -> job $jobid"
    done
  done
done
