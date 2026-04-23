#!/bin/bash
# Submit image generation with alpha=1.0 for steering experiments 4-7.
# Exps 8-11 are handled separately by submit_experiments_ziyang_alpha1.sh
# (ziyang profile). New images land in steered_alpha_1.0/ subfolders
# alongside existing steered_alpha_2.0/ images (which are NOT modified).
#
# Array ranges below reflect shards ALREADY completed for alpha=1.0:
#   Exp 4:  array 0-2 done  → submit 3-7
#   Exp 5:  array 0   done  → submit 1-7
#   Exp 6:  array 0   done  → submit 1-7
#   Exp 7:  array 0   done  → submit 1-7
# (run_experiment.py skips images that already exist, so resubmitting a
# completed shard is also safe — this is purely to avoid wasted queue time.)
#
# Usage:
#   bash submit_experiments_alpha1.sh                # Default: 4 5 6 7
#   bash submit_experiments_alpha1.sh 4 6            # Specific experiments

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p ../logs

# Per-experiment remaining array ranges (computed from disk state).
declare -A ARRAY_RANGE=(
    [4]="3-7"
    [5]="1-7"
    [6]="1-7"
    [7]="1-7"
)

# Parse experiment IDs (default: steering experiments assigned to ydai)
if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(4 5 6 7)
fi

echo "============================================"
echo "Submitting alpha=1.0 runs for experiments: ${EXP_IDS[*]}"
echo "Seeds controlled by config.py; alpha forced to 1.0 via --alphas CLI."
echo "============================================"

for EXP_ID in "${EXP_IDS[@]}"; do
    RANGE="${ARRAY_RANGE[$EXP_ID]:-0-7}"
    JOB_GEN=$(sbatch \
        --job-name="exp${EXP_ID}-a1-imggen" \
        --array=${RANGE} \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment_alpha1.slurm | awk '{print $4}')
    echo "Exp ${EXP_ID} (alpha=1.0, array=${RANGE}) image gen: Job ${JOB_GEN}"
done

echo "============================================"
echo "All image gen jobs submitted! Monitor with: squeue -u \$USER"
echo "After generation completes:"
echo "  1. Rebuild manifests:"
echo "       cd ..  &&  python rebuild_manifests.py 4 5 6 7"
echo "  2. Run evaluation (resumable, only new rows will be scored):"
echo "       cd ..  &&  OPENROUTER_API_KEY=\$OPENROUTER_API_KEY python evaluate_all.py --exp-id <N>"
echo "============================================"
