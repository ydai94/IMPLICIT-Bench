#!/bin/bash
#SBATCH --job-name=exp-zy-a1-resume
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100-short
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/exp_imggen_zy_a1_resume_%A_%a.out
#SBATCH --error=logs/exp_imggen_zy_a1_resume_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# alpha=1.0 image generation RESUME pass for ziyang experiments 8-11 on
# gpu-a100-short with a 1-hour walltime. Both prior passes (gpu-a100 and
# gpu-a100-short 4h) timed out leaving the tail of each shard's case list
# ungenerated; run_experiment.py skips images already on disk, so 1h is
# enough to mop up the remainder.
#
# Mirrors the ziyang-profile env activation (PYTHONNOUSERSITE, absolute
# conda env path, sanity check). 16-shard partitioning matches
# run_experiment_ziyang_alpha1*.slurm so already-generated images are
# skipped on disk.
#
# Usage (submitted by submit_experiments_ziyang_alpha1_resume.sh):
#   EXPERIMENT_ID=8 sbatch --array=12-15 run_experiment_ziyang_alpha1_resume.slurm
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
TOTAL_SHARDS=16
SHARD=${SLURM_ARRAY_TASK_ID}

EXP_ID=${EXPERIMENT_ID:-8}

echo "========================================"
echo "Experiment: ${EXP_ID} (alpha=1.0, ziyang, gpu-a100-short, 1h resume)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Shard: ${SHARD}/${TOTAL_SHARDS}"
echo "User: $(whoami)"
echo "python = $(which python)"
echo "========================================"

cd ${SCRIPT_DIR}

python run_experiment.py \
    --exp-id ${EXP_ID} \
    --gpu 0 \
    --shard ${SHARD} \
    --num-shards ${TOTAL_SHARDS} \
    --alphas 1.0

echo "Shard ${SHARD} done for experiment ${EXP_ID} (alpha=1.0, resume)."
