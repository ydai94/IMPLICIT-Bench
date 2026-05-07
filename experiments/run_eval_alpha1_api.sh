#!/bin/bash
# OpenRouter API Qwen3-VL bias + alignment evaluation on alpha=1.0 images for
# steering experiments (4-11). Uses no GPUs (network-bound, parallel workers).
#
# Reads from the stable manifest_alpha1.csv snapshot per experiment and writes
# to the same alpha1 result files used by the local wrapper:
#   cache/eval_results/exp_NN_alpha1_eval.{jsonl,csv}        (bias)
#   cache/eval_results/exp_NN_alpha1_alignment.{jsonl,csv}   (alignment)
#
# The API scripts (evaluate_all.py, evaluate_alignment.py) resume on
# (case_id, alpha, seed), so this wrapper picks up wherever the local wrapper
# left off and skips already-scored rows.
#
# Usage (run inside tmux/screen):
#   bash run_eval_alpha1_api.sh
#
# Subset overrides:
#   EXP_IDS="9 10 11"        bash run_eval_alpha1_api.sh
#   METRICS="bias"           bash run_eval_alpha1_api.sh
#   WORKERS=8                bash run_eval_alpha1_api.sh

set -euo pipefail

# Activate your Python/conda environment here.

SCRIPT_DIR=/path/to/stereoimage/experiments
RESULTS_DIR=/path/to/stereoimage/cache/eval_results
LOG_DIR=${SCRIPT_DIR}/logs
mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

EXP_IDS="${EXP_IDS:-4 5 6 7 8 9 10 11}"
METRICS="${METRICS:-bias alignment}"
MANIFEST_NAME="${MANIFEST_NAME:-manifest_alpha1.csv}"
WORKERS="${WORKERS:-16}"

# API scripts load OPENROUTER_API_KEY from the env or fall back to the
# stereoset .env file. Surface a clear error here if neither is set.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f /path/to/data/stereoset/backup/stereoset-augment/.env ]; then
    echo "ERROR: OPENROUTER_API_KEY not set and .env fallback missing" >&2
    exit 1
fi

echo "================================================================"
echo "API eval: exps=[${EXP_IDS}] metrics=[${METRICS}] workers=${WORKERS}"
echo "Manifest: ${MANIFEST_NAME}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

echo
echo "---- Refreshing alpha=1.0 manifests for ${EXP_IDS} ----"
python -u build_alpha1_manifest.py ${EXP_IDS}

for EXP in ${EXP_IDS}; do
    EXP2=$(printf '%02d' "${EXP}")
    EXP_DIR=$(python -c "from config import exp_output_dir; print(exp_output_dir(${EXP}))")
    MANIFEST_PATH="${EXP_DIR}/${MANIFEST_NAME}"

    for METRIC in ${METRICS}; do
        case "${METRIC}" in
            bias)
                SCRIPT=evaluate_all.py
                TAG=alpha1_eval
                ;;
            alignment)
                SCRIPT=evaluate_alignment.py
                TAG=alpha1_alignment
                ;;
            *) echo "Unknown metric: ${METRIC}" >&2; exit 1 ;;
        esac

        OUT_JSONL="${RESULTS_DIR}/exp_${EXP2}_${TAG}.jsonl"
        LOG="${LOG_DIR}/eval_api_alpha1_exp_${EXP2}_${METRIC}.log"
        PHASE_START=$(date +%s)

        echo
        echo "---- exp_${EXP2}_${METRIC} (API) starting at $(date '+%H:%M:%S') ----"
        echo "  manifest: ${MANIFEST_PATH}"
        echo "  output:   ${OUT_JSONL}"
        echo "  log:      ${LOG}"

        python -u "${SCRIPT}" \
            --exp-id "${EXP}" \
            --manifest "${MANIFEST_PATH}" \
            --output-jsonl "${OUT_JSONL}" \
            --workers "${WORKERS}" \
            > "${LOG}" 2>&1

        PHASE_END=$(date +%s)
        ELAPSED=$((PHASE_END - PHASE_START))
        printf "  exp_%s_%s done in %02dh%02dm%02ds\n" \
            "${EXP2}" "${METRIC}" \
            $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
    done
done

echo
echo "================================================================"
echo "All API phases complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs: cache/eval_results/exp_NN_alpha1_eval.{jsonl,csv}"
echo "         cache/eval_results/exp_NN_alpha1_alignment.{jsonl,csv}"
echo "================================================================"
