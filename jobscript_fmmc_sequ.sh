#!/bin/bash
#SBATCH --job-name=pokec_fmmc_small_sequ
#SBATCH --output=out_err/pokec_small_sequ_fmmc_%A_%a.out
#SBATCH --error=out_err/pokec_small_sequ_fmmc_%A_%a.err
#SBATCH --partition=general1
#SBATCH --nodes=1  # One node per task
#SBATCH --ntasks=1  # One task per node
#SBATCH --cpus-per-task=80  # 80 CPUs per task (using all logical cores on the node)
#SBATCH --mem=188G  # Memory per node
#SBATCH --array=0-8  # Array of 2 jobs (0, 1)
#SBATCH --time=144:00:00  # The maximum time for all tasks

# Load necessary modules (if required)
module load intel/oneapi/2023.2.0

# Set the MOSEK license path
export MOSEKLM_LICENSE_FILE=/home/pyllm/forth/mosek/mosek.lic

# Run the experiment corresponding to the array task ID using micromamba run
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID
    
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID
    
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID
    
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID
    
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID

elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID

elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID

elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
    srun --time=120:00:00 /scratch/pyllm/forth/bin/micromamba run -p /scratch/pyllm/forth/envs/fair_graph310_fmmc python /scratch/pyllm/forth/fair_nodesampling/run_scripts/fmmc_run_experiment_sequ.py $SLURM_ARRAY_TASK_ID

fi