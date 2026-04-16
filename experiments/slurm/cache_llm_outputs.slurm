#!/bin/bash
#SBATCH --job-name=cache-llm
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/cache_llm_%A_%a.out
#SBATCH --error=logs/cache_llm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Phase 1c: Cache LLM outputs (prompt rewrites + pair generation)
# Uses 4 array tasks, each processing 1/4 of the data
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

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shard: ${SLURM_ARRAY_TASK_ID} of ${NUM_SHARDS}"
echo "========================================"

cd ${SCRIPT_DIR}
python cache_llm_outputs.py \
    --gpu 0 \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --num-shards ${NUM_SHARDS}

echo "Shard ${SLURM_ARRAY_TASK_ID} complete."
