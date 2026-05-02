#!/bin/bash
# Resume alpha=1.0 image generation for experiments 4-7 on gpu-a100-short
# with a 1-hour walltime. Both prior passes (gpu-a100 5h and gpu-a100-short
# 4h) timed out leaving ~9% of PNGs ungenerated; run_experiment.py skips
# images already on disk, so this run only does the tail of each shard.
#
# Per-experiment array ranges below were computed from disk on
# 2026-05-02 (only shards with missing alpha=1.0 PNGs are submitted):
#   Exp 4: shards 10-15 missing (~50 imgs each)
#   Exp 5: shards  8-15 missing (~46-56 imgs each)
#   Exp 6: shards  6-15 missing (~46-52 imgs each)
#   Exp 7: shards  6-15 missing (~48-53 imgs each)
#
# Usage:
#   cd /data/gpfs/projects/punim2888/stereoimage/experiments/slurm
#   bash submit_experiments_alpha1_resume.sh           # all of 4 5 6 7
#   bash submit_experiments_alpha1_resume.sh 4 6       # specific exps

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p logs

declare -A ARRAY_RANGE=(
    [4]="10-15"
    [5]="8-15"
    [6]="6-15"
    [7]="6-15"
)

if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(4 5 6 7)
fi

# Validate: positional args must be integers in {4,5,6,7}
for EXP_ID in "${EXP_IDS[@]}"; do
    if ! [[ "$EXP_ID" =~ ^(4|5|6|7)$ ]]; then
        echo "ERROR: '$EXP_ID' is not a valid experiment ID."
        echo "Usage: bash $(basename "$0") [EXP_ID ...]   (each EXP_ID in 4,5,6,7)"
        echo "Examples:"
        echo "  bash $(basename "$0")          # submit all of 4 5 6 7"
        echo "  bash $(basename "$0") 4 6      # submit only exp 4 and exp 6"
        exit 1
    fi
done

echo "============================================"
echo "Submitting alpha=1.0 RESUME runs for: ${EXP_IDS[*]}"
echo "Partition: gpu-a100-short (1h, 1 GPU/job)"
echo "User: $(whoami)"
echo "============================================"

for EXP_ID in "${EXP_IDS[@]}"; do
    RANGE="${ARRAY_RANGE[$EXP_ID]:-0-15}"
    JOB_GEN=$(sbatch \
        --job-name="exp${EXP_ID}-resume" \
        --array=${RANGE} \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment_alpha1_resume.slurm | awk '{print $4}')
    echo "Exp ${EXP_ID} (alpha=1.0, array=${RANGE}) image gen: Job ${JOB_GEN}"
done

echo "============================================"
echo "Submitted. Monitor with: squeue -u \$(whoami)"
echo "After generation completes:"
echo "  1. Rebuild manifests:"
echo "       cd ..  &&  python rebuild_manifests.py 4 5 6 7"
echo "  2. Run evaluation (resumable, only new rows will be scored):"
echo "       cd ..  &&  OPENROUTER_API_KEY=\$OPENROUTER_API_KEY python evaluate_all.py --exp-id <N>"
echo "============================================"
