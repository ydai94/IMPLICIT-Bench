#!/bin/bash
#SBATCH --job-name=cache-emb
#SBATCH --account=punim2888
#SBATCH --partition=gpu-a100
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cache_embeddings_%j.out
#SBATCH --error=logs/cache_embeddings_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yue.dai.3@student.unimelb.edu.au

# ============================================================
# Phase 1a+1b: Cache embeddings and extract KG via retrieval
# ============================================================

module purge
module load Anaconda3/2024.02-1
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2888/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim2888/anaconda3/pkgs
eval "$(conda shell.bash hook)"
conda activate videoGen
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh

SCRIPT_DIR=/data/gpfs/projects/punim2888/stereoimage/experiments
mkdir -p ${SCRIPT_DIR}/logs

echo "========================================"
echo "Phase 1a: Caching embeddings"
echo "========================================"
cd ${SCRIPT_DIR}
python cache_embeddings.py --gpu 0 --batch-size 32

echo "========================================"
echo "Phase 1b: Extracting KG via retrieval"
echo "========================================"
python cache_kg_extraction.py

echo "Done!"
