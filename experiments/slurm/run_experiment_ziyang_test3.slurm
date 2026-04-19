#!/bin/bash
#SBATCH --job-name=ziyang-test2
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test3_%A.out
#SBATCH --error=logs/test3_%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Quick test: run exp 1 on a small sample to confirm full pipeline works.
# 1 GPU, 5 min, 2 cases. Skips images that already exist.
#
# Submit:
#   sbatch run_experiment_ziyang_test3.slurm
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH

conda activate /data/gpfs/projects/punim2888/anaconda3/envs/videoGen
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh


echo "========================================"
echo "User: $(whoami)"
echo "Job: ${SLURM_JOB_ID}"
echo "python = $(which python)"
echo "========================================"

# Sanity check — fail loudly if env didn't activate
python -c "import sys; assert 'videoGen' in sys.executable, f'WRONG ENV: {sys.executable}'"

cd /data/gpfs/projects/punim2888/stereoimage/experiments

# Run exp 1 on first 2 cases only (skips already-generated images)
python run_experiment.py --exp-id 1 --gpu 0 --sample 2

echo "========================================"
echo "Test passed."
