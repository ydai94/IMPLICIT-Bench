#!/bin/bash
# Local Qwen-VL bias + alignment evaluation on alpha=1.0 images for steering
# experiments (4-11). Uses 2 GPUs on the current host (no SLURM).
#
# For each (exp, metric) phase, spawns two python shards in parallel
# (one per GPU), waits for both, then merges shards into the canonical
# cache/eval_results/exp_NN_<metric>_local.{jsonl,csv} files.
#
# Idempotent / resumable: shard JSONLs skip already-scored
# (case_id, alpha, seed) keys, so re-running picks up newly-generated
# images after rebuild_manifests.py is re-run.
#
# Usage (run inside tmux/screen, will take many hours):
#   bash run_eval_alpha1_local.sh
#
# Subset overrides:
#   EXP_IDS="4 5"           bash run_eval_alpha1_local.sh
#   METRICS="bias"          bash run_eval_alpha1_local.sh
#   EXP_IDS="4" METRICS="alignment" bash run_eval_alpha1_local.sh

set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
source /apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate videoGen

SCRIPT_DIR=/path/to/stereoimage/experiments
LOG_DIR=${SCRIPT_DIR}/logs
mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

EXP_IDS="${EXP_IDS:-4 5 6 7 8 9 10 11}"
METRICS="${METRICS:-bias alignment}"
ALPHA="${ALPHA:-1.0}"
MANIFEST_NAME="${MANIFEST_NAME:-manifest_alpha1.csv}"
NUM_SHARDS=2

echo "================================================================"
echo "Local 2-GPU eval: exps=[${EXP_IDS}] metrics=[${METRICS}] alpha=${ALPHA}"
echo "Manifest: ${MANIFEST_NAME} (stable disk-walk snapshot)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo
echo "---- Building stable alpha=1.0 manifests for ${EXP_IDS} ----"
python -u build_alpha1_manifest.py ${EXP_IDS}

for EXP in ${EXP_IDS}; do
    for METRIC in ${METRICS}; do
        case "${METRIC}" in
            bias)      SCRIPT=evaluate_bias_local.py ;;
            alignment) SCRIPT=evaluate_alignment_local.py ;;
            *) echo "Unknown metric: ${METRIC}" >&2; exit 1 ;;
        esac

        EXP2=$(printf '%02d' "${EXP}")
        PHASE_TAG="exp_${EXP2}_${METRIC}"
        PHASE_START=$(date +%s)

        echo
        echo "---- ${PHASE_TAG} (alpha=${ALPHA}) starting at $(date '+%H:%M:%S') ----"

        TAG="alpha1_${METRIC/bias/eval}"  # bias->alpha1_eval, alignment->alpha1_alignment

        for GPU in 0 1; do
            LOG="${LOG_DIR}/eval_local_${PHASE_TAG}_shard_${GPU}.log"
            echo "  GPU ${GPU} -> shard ${GPU}/${NUM_SHARDS}, log: ${LOG}"
            python -u "${SCRIPT}" \
                --gpu "${GPU}" \
                --shard "${GPU}" \
                --num-shards "${NUM_SHARDS}" \
                --exp-id "${EXP}" \
                --manifest "${MANIFEST_NAME}" \
                --alpha "${ALPHA}" \
                --name-tag "${TAG}" \
                > "${LOG}" 2>&1 &
        done

        wait

        echo "  merging shards..."
        python -u "${SCRIPT}" --merge-only --exp-id "${EXP}" --name-tag "${TAG}" \
            >> "${LOG_DIR}/eval_local_${PHASE_TAG}_merge.log" 2>&1

        PHASE_END=$(date +%s)
        ELAPSED=$((PHASE_END - PHASE_START))
        printf "  %s done in %02dh%02dm%02ds\n" \
            "${PHASE_TAG}" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
    done
done

echo
echo "================================================================"
echo "All phases complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs: cache/eval_results/exp_NN_alpha1_eval.{jsonl,csv}"
echo "         cache/eval_results/exp_NN_alpha1_alignment.{jsonl,csv}"
echo "================================================================"
