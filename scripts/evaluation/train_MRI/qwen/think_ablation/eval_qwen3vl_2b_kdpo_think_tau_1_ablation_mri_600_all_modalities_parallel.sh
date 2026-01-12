#!/bin/bash

# Evaluate Qwen3-VL-2B (SFT mode) on all 8 modalities using 8 GPUs in parallel
# Training: MRI (600 samples)
# Testing: 8 modalities × 300 samples each
# Each GPU processes 1 modality
# Example: bash scripts/evaluation/train_MRI/qwen/think_ablation/eval_qwen3vl_2b_kdpo_think_tau_1_ablation_mri_600_all_modalities_parallel.sh

MODEL_PATH="/mnt/task_runtime/output/MM-GOLD/Qwen3-VL-2B-KDPO-MRI-600-think-tau-1"
BATCH_SIZE=16
MODE="think"

EVAL_JSON_DIR="/mnt/task_runtime/data/omni_med_vqa_processed_sampled/eval_json"
IMAGE_FOLDER="/mnt/task_runtime/data/OmniMedVQA/OmniMedVQA"

if [ "${MODE}" == "think" ]; then
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa.py"
else
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa_nothink.py"
fi

declare -a MODALITIES=(
    "Fundus:Fundus_test_300.json:0"
    "Ultrasound:Ultrasound_test_300.json:1"
    "Dermoscopy:Dermoscopy_test_300.json:2"
    "OCT:OCT_test_300.json:3"
    "Microscopy:Microscopy_test_300.json:4"
    "CT:CT_test_300.json:5"
    "XRay:X-Ray_test_300.json:6"
    "MRI:MRI_test_300.json:7"
)

# Find all checkpoints under MODEL_PATH (sorted)
CHECKPOINTS=()
while IFS= read -r -d $'\0' chkpt; do
    CHECKPOINTS+=("$chkpt")
done < <(find "${MODEL_PATH}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tr '\n' '\0')

# If no checkpoints found, fallback to just the MODEL_PATH itself for evaluation
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    CHECKPOINTS=("${MODEL_PATH}")
fi

for CHKPT_PATH in "${CHECKPOINTS[@]}"; do
    # Extract checkpoint name or use base dir
    CHKPT_BASENAME=$(basename "${CHKPT_PATH}")
    if [[ "${CHKPT_BASENAME}" == checkpoint-* ]]; then
        MODEL_NAME="Qwen3-VL-2B-KDPO-MRI-600-think-tau-1_${CHKPT_BASENAME}"
    else
        MODEL_NAME="Qwen3-VL-2B-KDPO-MRI-600-think-tau-1"
    fi
    BASE_OUTPUT_DIR="/mnt/task_runtime/results/${MODEL_NAME}"
    mkdir -p "${BASE_OUTPUT_DIR}"

    echo "=========================================="
    echo "Evaluating ${MODEL_NAME} (${MODE} mode)"
    echo "Model: ${CHKPT_PATH}"
    echo "Output: ${BASE_OUTPUT_DIR}"
    echo "Using 8 GPUs in parallel"
    echo "=========================================="

    # Function to check if result file exists and is complete
    check_result_complete() {
        local output_path=$1
        local expected_samples=300

        if [ ! -f "${output_path}" ]; then
            return 1  # File doesn't exist
        fi

        local sample_count
        sample_count=$(python3 -c "
import json
try:
    with open('${output_path}', 'r') as f:
        data = json.load(f)
    results = data['results'] if isinstance(data, dict) and 'results' in data else []
    print(len(results))
except Exception:
    print(0)
" 2>/dev/null)

        if [ "${sample_count}" -eq "${expected_samples}" ]; then
            return 0
        else
            return 1
        fi
    }

    # Evaluation per modality (with retry for OOM)
    evaluate_modality() {
        local modality=$1 json_file=$2 gpu_id=$3 model_path=$4 model_name=$5 base_output_dir=$6
        local PROMPT="${EVAL_JSON_DIR}/${json_file}"
        local OUT="${base_output_dir}/test_${modality}_${model_name}_results.json"
        local LOG="/tmp/eval_${modality}_gpu${gpu_id}.log"

        if check_result_complete "${OUT}"; then
            echo "[GPU ${gpu_id}] ⏭️  ${modality} already complete"
            return 0
        fi
        python -c "import torch; torch.cuda.set_device(${gpu_id}); torch.cuda.empty_cache()" 2>/dev/null || true

        local bsz=$BATCH_SIZE; local try=0; local max_try=2
        while [ "$try" -le "$max_try" ]; do
            if [ "$try" -gt 0 ]; then
                bsz=$((bsz/2)); ((bsz<1)) && bsz=1
                echo "[GPU ${gpu_id}] Retrying ${modality} batch_size=${bsz}"
                python -c "import torch; torch.cuda.set_device(${gpu_id}); torch.cuda.empty_cache()" 2>/dev/null || true
                sleep 2
            fi
            if python "${EVAL_SCRIPT}" \
                --model_path "${model_path}" \
                --batch_size "${bsz}" \
                --output_path "${OUT}" \
                --prompt_path "${PROMPT}" \
                --image_folder "${IMAGE_FOLDER}" \
                --gpu_id "${gpu_id}" \
                > "${LOG}" 2>&1; then
                echo "[GPU ${gpu_id}] ✓ ${modality} done (batch_size=${bsz})"
                return 0
            else
                if grep -qi "out of memory\|CUDA out of memory" "${LOG}" 2>/dev/null; then
                    ((try++))
                    if [ "$try" -gt "$max_try" ]; then
                        echo "[GPU ${gpu_id}] OOM: ${modality} failed after $max_try retries"
                        return 1
                    fi
                else
                    echo "[GPU ${gpu_id}] ✗ ${modality} failed"
                    tail -5 "${LOG}" | sed 's/^/   /'
                    return 1
                fi
            fi
        done
    }

    export -f evaluate_modality
    export -f check_result_complete
    export BATCH_SIZE MODE EVAL_SCRIPT EVAL_JSON_DIR IMAGE_FOLDER

    # Parallel evaluation per modality for this checkpoint
    for entry in "${MODALITIES[@]}"; do
        IFS=':' read -r mod file gpu <<< "$entry"
        (
            if [ ! -f "${EVAL_JSON_DIR}/${file}" ]; then
                echo "[GPU $gpu] ERROR: missing ${file}"
                exit 1
            fi
            evaluate_modality "${mod}" "${file}" "${gpu}" "${CHKPT_PATH}" "${MODEL_NAME}" "${BASE_OUTPUT_DIR}"
        ) &
    done

    wait
    echo "=========================================="
    echo "All evaluations for ${MODEL_NAME} attempted."
    echo "Results saved to: ${BASE_OUTPUT_DIR}"
    echo "=========================================="

done

