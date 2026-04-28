#!/bin/bash
#SBATCH --job-name=exp-imggen-short
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100-short
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/exp_imggen_short_%A_%a.out
#SBATCH --error=logs/exp_imggen_short_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# alpha=1.0 image generation on gpu-a100-short (4h cap, 1 GPU/job).
# 16-shard partitioning matches run_experiment_alpha1.slurm so already-
# generated images are skipped on disk. Safe to run alongside any
# pending gpu-a100 alpha=1.0 jobs (file-exists check guards each seed).
#
# Usage (defaults to exp 5, all 16 shards):
#   sbatch --array=0-15 run_experiment_alpha1_short.slurm
#   EXPERIMENT_ID=5 sbatch --array=8-15 run_experiment_alpha1_short.slurm
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
eval "$(conda shell.bash hook)"
conda activate videoGen
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments
TOTAL_SHARDS=16
SHARD=${SLURM_ARRAY_TASK_ID}

EXP_ID=${EXPERIMENT_ID:-5}

echo "========================================"
echo "Experiment: ${EXP_ID} (alpha=1.0, gpu-a100-short)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shard: ${SHARD}/${TOTAL_SHARDS}"
echo "python = $(which python)"
echo "========================================"

cd ${SCRIPT_DIR}

python run_experiment.py \
    --exp-id ${EXP_ID} \
    --gpu 0 \
    --shard ${SHARD} \
    --num-shards ${TOTAL_SHARDS} \
    --alphas 1.0

echo "Shard ${SHARD} done for experiment ${EXP_ID} (alpha=1.0)."
