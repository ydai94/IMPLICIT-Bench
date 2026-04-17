#!/bin/bash
#SBATCH --job-name=exp-imggen
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/exp_imggen_%A_%a.out
#SBATCH --error=logs/exp_imggen_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Image generation for ziyang — experiments 1, 2, 3
# Skips already-generated images automatically.
#
# Usage (submitted by submit_experiments_ziyang.sh):
#   EXPERIMENT_ID=1 sbatch --array=4-7 run_experiment_ziyang.slurm
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
eval "$(conda shell.bash hook)"
conda activate videoGen
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments
NUM_GPUS_PER_NODE=2
NUM_ARRAY_TASKS=8
TOTAL_SHARDS=$((NUM_ARRAY_TASKS * NUM_GPUS_PER_NODE))
BASE_SHARD=$((SLURM_ARRAY_TASK_ID * NUM_GPUS_PER_NODE))

# Default to experiment 1 if not set
EXP_ID=${EXPERIMENT_ID:-1}

echo "========================================"
echo "Experiment: ${EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shards: ${BASE_SHARD} and $((BASE_SHARD + 1)) of ${TOTAL_SHARDS}"
echo "User: $(whoami)"
echo "========================================"

cd ${SCRIPT_DIR}

for GPU_ID in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
    SHARD=$((BASE_SHARD + GPU_ID))
    echo "Starting shard ${SHARD}/${TOTAL_SHARDS} on GPU ${GPU_ID}"

    python run_experiment.py \
        --exp-id ${EXP_ID} \
        --gpu ${GPU_ID} \
        --shard ${SHARD} \
        --num-shards ${TOTAL_SHARDS} \
        &
done

wait
echo "All shards complete for experiment ${EXP_ID}."
