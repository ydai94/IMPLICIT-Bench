#!/bin/bash
# Submit image generation + evaluation for experiments.
# Cache phases are already complete — this skips directly to Phase 2+3.
#
# Usage:
#   bash submit_experiments.sh                        # All 11 experiments (1-11)
#   bash submit_experiments.sh 1 2 3 4 5 6 7 8 9 10 11  # Specific experiments

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p ../logs

# Parse experiment IDs (default: all except 0 which is already done)
if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(1 2 3 4 5 6 7 8 9 10 11)
fi

echo "============================================"
echo "Submitting experiments: ${EXP_IDS[*]}"
echo "Alphas and seeds are controlled by config.py"
echo "============================================"

for EXP_ID in "${EXP_IDS[@]}"; do
    # Image generation only (evaluation uses OpenRouter API separately)
    JOB_GEN=$(sbatch \
        --job-name="exp${EXP_ID}-imggen" \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment.slurm | awk '{print $4}')
    echo "Exp ${EXP_ID} image gen: Job ${JOB_GEN}"
done

echo "============================================"
echo "All image gen jobs submitted! Monitor with: squeue -u \$USER"
echo "After generation completes, run evaluation via OpenRouter API:"
echo "  cd ../  &&  OPENROUTER_API_KEY=\$OPENROUTER_API_KEY python evaluate_all.py --exp-id <N>"
echo "============================================"
