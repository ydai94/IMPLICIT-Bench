#!/bin/bash
#SBATCH --job-name=exp-imggen-zy-a1-test
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=logs/exp_imggen_zy_a1_test_%A_%a.out
#SBATCH --error=logs/exp_imggen_zy_a1_test_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# 5-minute smoke test of the ziyang alpha=1.0 SLURM pipeline.
# Minimal resources (1 GPU, 1 shard, 2 cases) so it schedules fast.
# Purpose: confirm env activation, GPU allocation, and model load.
# NOT a throughput test.
#
# Submit directly with sbatch — no wrapper needed:
#   cd /data/gpfs/projects/punim2888/stereoimage/experiments/slurm
#   sbatch run_experiment_ziyang_alpha1_test.slurm
#
# Override the experiment id if desired:
#   EXPERIMENT_ID=10 sbatch run_experiment_ziyang_alpha1_test.slurm
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
python -c "from diffusers import QwenImagePipeline; print('diffusers OK')"

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments

# Default to experiment 8 if not set
EXP_ID=${EXPERIMENT_ID:-8}

echo "========================================"
echo "Experiment: ${EXP_ID} (alpha=1.0, ziyang TEST 5min, 1 GPU)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "User: $(whoami)"
echo "python = $(which python)"
nvidia-smi || true
echo "========================================"

cd ${SCRIPT_DIR}

# Single shard on the only GPU; --sample 2 keeps the run short.
python run_experiment.py \
    --exp-id ${EXP_ID} \
    --gpu 0 \
    --shard 0 \
    --num-shards 1 \
    --alphas 1.0 \
    --sample 2

echo "Smoke test complete for experiment ${EXP_ID} (alpha=1.0)."
