#!/bin/bash
# Submit image generation with alpha=1.0 for steering experiments 8, 9, 10, 11.
# Intended for ziyang on punim2888 (mirrors submit_experiments_ziyang.sh).
# Output lands in steered_alpha_1.0/ subfolders alongside existing
# steered_alpha_2.0/ images (which are NOT modified).
#
# Array ranges below reflect shards ALREADY completed for alpha=1.0:
#   Exp 8:  array 0     done  → submit 1-7
#   Exp 9:  array 0-1   done  → submit 2-7
#   Exp 10: array 0     done  → submit 1-7
#   Exp 11: array 0     done  → submit 1-7
# (run_experiment.py skips images that already exist, so resubmitting a
# completed shard is also safe — this is purely to avoid wasted queue time.)
#
# Usage:
#   cd /data/gpfs/projects/punim2888/stereoimage/experiments/slurm
#   bash submit_experiments_ziyang_alpha1.sh                # All of 8-11
#   bash submit_experiments_ziyang_alpha1.sh 8 10           # Specific exps

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p logs

declare -A ARRAY_RANGE=(
    [8]="1-7"
    [9]="2-7"
    [10]="1-7"
    [11]="1-7"
)

# Parse experiment IDs (default: steering exps assigned to ziyang)
if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(8 9 10 11)
fi

echo "============================================"
echo "Submitting alpha=1.0 runs for experiments: ${EXP_IDS[*]}"
echo "User: $(whoami)"
echo "Profile: ziyang (PYTHONNOUSERSITE=1, absolute-path conda activate)"
echo "============================================"

for EXP_ID in "${EXP_IDS[@]}"; do
    RANGE="${ARRAY_RANGE[$EXP_ID]:-0-7}"
    JOB_GEN=$(sbatch \
        --job-name="exp${EXP_ID}-a1-zy" \
        --array=${RANGE} \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment_ziyang_alpha1.slurm | awk '{print $4}')
    echo "Exp ${EXP_ID} (alpha=1.0, array=${RANGE}) image gen: Job ${JOB_GEN}"
done

echo "============================================"
echo "Each array task: 1 node, 2 A100 GPUs, ~5h wall time"
echo "Monitor with: squeue -u \$(whoami)"
echo "After generation completes:"
echo "  1. Rebuild manifests:"
echo "       cd ..  &&  python rebuild_manifests.py 8 9 10 11"
echo "  2. Run evaluation (resumable):"
echo "       cd ..  &&  OPENROUTER_API_KEY=\$OPENROUTER_API_KEY python evaluate_all.py --exp-id <N>"
echo "============================================"
