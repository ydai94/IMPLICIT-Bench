#!/bin/bash
#SBATCH --job-name=exp-imggen-zy-a1
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/exp_imggen_zy_a1_%A_%a.out
#SBATCH --error=logs/exp_imggen_zy_a1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Image generation for ziyang — steering experiments 8-11 with alpha=1.0
# Mirrors run_experiment_ziyang.slurm layout (PYTHONNOUSERSITE,
# explicit env-activation sanity check) but forces --alphas 1.0.
# Output lands in <exp>/<case>/steered_alpha_1.0/seed_*.png
# and does NOT disturb existing steered_alpha_2.0/ images.
#
# Usage (submitted by submit_experiments_ziyang_alpha1.sh):
#   EXPERIMENT_ID=8 sbatch --array=0-7 run_experiment_ziyang_alpha1.slurm
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH

source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/gpfs/projects/punim2888/anaconda3/envs/videoGen

# Sanity check — fail loudly if env didn't activate
python -c "import sys; assert 'videoGen' in sys.executable, f'WRONG ENV: {sys.executable}'"

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments
NUM_GPUS_PER_NODE=2
NUM_ARRAY_TASKS=8
TOTAL_SHARDS=$((NUM_ARRAY_TASKS * NUM_GPUS_PER_NODE))
BASE_SHARD=$((SLURM_ARRAY_TASK_ID * NUM_GPUS_PER_NODE))

# Default to experiment 8 if not set
EXP_ID=${EXPERIMENT_ID:-8}

echo "========================================"
echo "Experiment: ${EXP_ID} (alpha=1.0, ziyang)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shards: ${BASE_SHARD} and $((BASE_SHARD + 1)) of ${TOTAL_SHARDS}"
echo "User: $(whoami)"
echo "python = $(which python)"
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
