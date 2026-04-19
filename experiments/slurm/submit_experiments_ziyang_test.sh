#!/bin/bash
# Submit remaining image generation tasks for experiments 1, 2, 3.
# For use by ziyang on punim2888.
#
# Based on completed jobs as of 2026-04-16:
#   Exp 1: tasks 0-3 done (2757/5493 images), tasks 4-7 remaining
#   Exp 2: tasks 0-2 done (2070/5493 images), tasks 3-7 remaining
#   Exp 3: tasks 0-3 done (2757/5493 images), tasks 4-7 remaining
#
# The Python script skips images that already exist on disk,
# so re-running completed shards is safe (they exit quickly).
#
# Usage:
#   cd /data/gpfs/projects/punim2888/stereoimage/experiments/slurm
#   bash submit_experiments_ziyang.sh

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p logs

echo "============================================"
echo "Submitting experiments 1, 2, 3 (remaining tasks only)"
echo "User: $(whoami)"
echo "============================================"

# Exp 1: tasks 4-7 remaining (4 array tasks × 2 GPUs = 8 shards)
JOB1=$(sbatch \
    --job-name="exp1-imggen-zy" \
    --array=4 \
    --export=ALL,EXPERIMENT_ID=1 \
    run_experiment_ziyang_test.slurm | awk '{print $4}')
echo "Exp 1 image gen (tasks 4-7): Job ${JOB1}"

# Exp 2: tasks 3-7 remaining (5 array tasks × 2 GPUs = 10 shards)
JOB2=$(sbatch \
    --job-name="exp2-imggen-zy" \
    --array=3 \
    --export=ALL,EXPERIMENT_ID=2 \
    run_experiment_ziyang_test.slurm | awk '{print $4}')
echo "Exp 2 image gen (tasks 3-7): Job ${JOB2}"

# Exp 3: tasks 4-7 remaining (4 array tasks × 2 GPUs = 8 shards)
JOB3=$(sbatch \
    --job-name="exp3-imggen-zy" \
    --array=4\
    --export=ALL,EXPERIMENT_ID=3 \
    run_experiment_ziyang_test.slurm | awk '{print $4}')
echo "Exp 3 image gen (tasks 4-7): Job ${JOB3}"

echo "============================================"
echo "Submitted 13 array tasks total (4 + 5 + 4)"
echo "Each task: 1 node, 2 A100 GPUs, ~5h wall time"
echo "Monitor with: squeue -u \$(whoami)"
echo "============================================"