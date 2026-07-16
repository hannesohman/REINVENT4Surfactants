#!/bin/env bash
#SBATCH -A NAISS2025-5-462          # find your project with the "projinfo" command
#SBATCH -p alvis                    # what partition to use (usually not needed)
#SBATCH -t 40:00:00                 # how long time it will take to run
# -C NOGPU
#SBATCH --gpus-per-node=A40:1
#SBATCH -J test_main                # the jobname (not needed)
#SBATCH -o {{OUTPUT_PATH}}            # name of the output file

# --nodes=1
# --ntasks=1
# --cpus-per-task=8

# Load modules
module purge
# module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# module load matplotlib/3.10.5-gfbf-2025b
# module load JupyterLab/4.4.9-GCCcore-14.3.0
# module load Python/3.13.5-GCCcore-14.3.0
# module load virtualenv/20.32.0-GCCcore-14.3.0

## THIS MAKES SURE PYTHON USES THE RIGHT LIBRARIES
module load virtualenv/20.32.0-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b
source venv/bin/activate

# Start low-priority filler
nice -n 19 python -u gpu_keepalive.py --mat_size 200 --sleep_time 0.7 &
FILLER_PID=$!
trap "kill $FILLER_PID 2>/dev/null" EXIT

python -u workflow/run_try.py \
 --config {{CONFIG_PATH}}