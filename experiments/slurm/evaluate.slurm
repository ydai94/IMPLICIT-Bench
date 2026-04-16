#!/bin/bash
#SBATCH --job-name=exp-eval
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/exp_eval_%A_%a.out
#SBATCH --error=logs/exp_eval_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Phase 3: Evaluate generated images with Qwen3-VL
# Set EXPERIMENT_ID before submitting:
#   EXPERIMENT_ID=0 sbatch evaluate.slurm
# Uses 4 array tasks for sharding
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
eval "$(conda shell.bash hook)"
conda activate videoGen
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments
NUM_SHARDS=4
EXP_ID=${EXPERIMENT_ID:-0}

echo "========================================"
echo "Evaluating Experiment: ${EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}, Shard: ${SLURM_ARRAY_TASK_ID} of ${NUM_SHARDS}"
echo "========================================"

cd ${SCRIPT_DIR}
python evaluate_all.py \
    --exp-id ${EXP_ID} \
    --gpu 0 \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --num-shards ${NUM_SHARDS}

echo "Shard ${SLURM_ARRAY_TASK_ID} complete."
