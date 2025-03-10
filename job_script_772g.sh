#!/usr/bin/env bash
#SBATCH --job-name=pokec_experiment
#SBATCH --output=out_err/pokec_experiment_%A_%a.out
#SBATCH --error=out_err/pokec_experiment_%A_%a.err
#SBATCH --partition=general1
#SBATCH --nodes=1                # One node per task
#SBATCH --ntasks=1               # One task per node
#SBATCH --cpus-per-task=80
#SBATCH --mem=750G
#SBATCH --array=3            # Array job: 1 experiment ['full']
#SBATCH --time=240:00:00         # Maximum walltime

# Load necessary modules (if required)
module load intel/oneapi/2023.2.0

# Get the task ID from the Slurm array or command-line argument.
TASK_ID=${SLURM_ARRAY_TASK_ID:-$1}
if [ -z "$TASK_ID" ]; then
  echo "Usage: $0 <task_id>"
  exit 1
fi

# Run the unified experiment pipeline using micromamba.
# This command assumes:
#   - Your Python environment is located at /scratch/pyllm/forth/envs/fair_graph310_fmmc
#   - The micromamba executable is at /scratch/pyllm/forth/bin/micromamba
#   - Your unified pipeline is at experiments/experiment_main.py
srun --time=240:00:00 \
  /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310 \
  python /scratch/pyllm/forth/fairgraphwalk-experiments/experiments/experiments_main.py \
  --config experiments/config.yml \
  --experiment_to_run both \
  --experiment_id "$TASK_ID" \
  --base_dir /scratch/pyllm/$USER/DATA

