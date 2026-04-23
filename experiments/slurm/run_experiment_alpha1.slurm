#!/bin/bash
#SBATCH --job-name=exp-imggen-a1
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --array=0-7
#SBATCH --output=logs/exp_imggen_a1_%A_%a.out
#SBATCH --error=logs/exp_imggen_a1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Image generation with alpha=1.0 for steering experiments (4-11)
# Set EXPERIMENT_ID before submitting:
#   EXPERIMENT_ID=4 sbatch run_experiment_alpha1.slurm
# Uses 8 array tasks x 2 GPUs = 16 shards
# Output lands in <exp>/<case>/steered_alpha_1.0/seed_*.png
# and does not touch existing steered_alpha_2.0/ images.
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

# Default to experiment 4 if not set
EXP_ID=${EXPERIMENT_ID:-4}

echo "========================================"
echo "Experiment: ${EXP_ID} (alpha=1.0)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shards: ${BASE_SHARD} and $((BASE_SHARD + 1)) of ${TOTAL_SHARDS}"
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
        --alphas 1.0 \
        &
done

wait
echo "All shards complete for experiment ${EXP_ID} (alpha=1.0)."
