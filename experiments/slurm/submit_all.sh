#!/bin/bash
# Submit the full experiment pipeline with job dependencies.
#
# Usage:
#   bash submit_all.sh              # Run all experiments sequentially
#   bash submit_all.sh 0 3 6 11     # Run specific experiment IDs only
#
# The pipeline is:
#   Phase 1: cache_embeddings + cache_kg_extraction
#   Phase 2: cache_llm_outputs (depends on Phase 1)
#   Phase 3: run_experiment for each exp_id (depends on Phase 2)
#   Phase 4: evaluate for each exp_id (depends on its Phase 3 job)

set -e

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SLURM_DIR}"
mkdir -p ../logs

# Parse experiment IDs (default: all 12)
if [ $# -gt 0 ]; then
    EXP_IDS=("$@")
else
    EXP_IDS=(0 1 2 3 4 5 6 7 8 9 10 11)
fi

echo "============================================"
echo "Submitting experiment pipeline"
echo "Experiments: ${EXP_IDS[*]}"
echo "============================================"

# Phase 1: Cache embeddings + KG extraction
JOB1=$(sbatch cache_embeddings.slurm | awk '{print $4}')
echo "Phase 1 (embeddings + KG extraction): Job ${JOB1}"

# Phase 2: Cache LLM outputs
JOB2=$(sbatch --dependency=afterok:${JOB1} cache_llm_outputs.slurm | awk '{print $4}')
echo "Phase 2 (LLM outputs): Job ${JOB2}"

# Phase 3+4: For each experiment, submit image gen then evaluation
for EXP_ID in "${EXP_IDS[@]}"; do
    JOB3=$(EXPERIMENT_ID=${EXP_ID} sbatch --dependency=afterok:${JOB2} \
        --job-name="exp${EXP_ID}-imggen" \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        run_experiment.slurm | awk '{print $4}')
    echo "Phase 3 (exp ${EXP_ID} image gen): Job ${JOB3}"

    JOB4=$(EXPERIMENT_ID=${EXP_ID} sbatch --dependency=afterok:${JOB3} \
        --job-name="exp${EXP_ID}-eval" \
        --export=ALL,EXPERIMENT_ID=${EXP_ID} \
        evaluate.slurm | awk '{print $4}')
    echo "Phase 4 (exp ${EXP_ID} evaluation): Job ${JOB4}"
done

echo "============================================"
echo "All jobs submitted! Monitor with: squeue -u \$USER"
echo "============================================"
