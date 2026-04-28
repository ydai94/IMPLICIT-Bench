#!/bin/bash
# Submit alpha=1.0 image generation for steering experiments 8-11 to
# gpu-a100-short (4h walltime, 1 GPU/job). Each array index = 1 shard
# of the 16-shard partition. Skip-on-disk in run_experiment.py guards
# against duplicate generation, so this is safe to run alongside any
# pending gpu-a100 alpha=1.0 jobs.
#
# Per-experiment array ranges below were computed from disk on
# 2026-04-28 (only shards with missing alpha=1.0 PNGs are submitted):
#   Exp 8:  shards 12-15 missing
#   Exp 9:  shards 8-15  missing
#   Exp 10: shards 6-15  missing
#   Exp 11: shards 6-15  missing
#
# Usage:
#   cd /data/gpfs/projects/punim2888/stereoimage/experiments/slurm
#   bash submit_experiments_ziyang_alpha1_short.sh                # All of 8-11
#   bash submit_experiments_ziyang_alpha1_short.sh 8 10           # Specific exps

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p logs

declare -A ARRAY_RANGE=(
    [8]="12-15"
    [9]="8-15"
    [10]="6-15"
    [11]="6-15"
)

if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(8 9 10 11)
fi

# Validate: positional args must be integers in {8,9,10,11}
for EXP_ID in "${EXP_IDS[@]}"; do
    if ! [[ "$EXP_ID" =~ ^(8|9|10|11)$ ]]; then
        echo "ERROR: '$EXP_ID' is not a valid experiment ID."
        echo "Usage: bash $(basename "$0") [EXP_ID ...]   (each EXP_ID in 8,9,10,11)"
        echo "Examples:"
        echo "  bash $(basename "$0")          # submit all of 8 9 10 11"
        echo "  bash $(basename "$0") 8 10     # submit only exp 8 and exp 10"
        exit 1
    fi
done

echo "============================================"
echo "Submitting alpha=1.0 short-partition runs for: ${EXP_IDS[*]}"
echo "Partition: gpu-a100-short (4h, 1 GPU/job)"
echo "User: $(whoami)"
echo "============================================"

for EXP_ID in "${EXP_IDS[@]}"; do
    RANGE="${ARRAY_RANGE[$EXP_ID]:-0-15}"
    JOB_GEN=$(sbatch \
        --job-name="exp${EXP_ID}-zy-short" \
        --array=${RANGE} \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment_ziyang_alpha1_short.slurm | awk '{print $4}')
    echo "Exp ${EXP_ID} (alpha=1.0, array=${RANGE}) image gen: Job ${JOB_GEN}"
done

echo "============================================"
echo "Submitted. Monitor with: squeue -u \$(whoami)"
echo "After generation completes:"
echo "  1. Rebuild manifests:"
echo "       cd ..  &&  python rebuild_manifests.py 8 9 10 11"
echo "  2. Run evaluation (resumable):"
echo "       cd ..  &&  OPENROUTER_API_KEY=\$OPENROUTER_API_KEY python evaluate_all.py --exp-id <N>"
echo "============================================"
