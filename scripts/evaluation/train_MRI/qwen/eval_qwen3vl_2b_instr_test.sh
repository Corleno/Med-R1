#!/bin/bash

# Evaluate Qwen3-VL-2B (Instruction mode) on specified modality
# Training: MRI (600 samples)
# Testing: 8 modalities × 300 samples each

# Usage:
#   bash eval_qwen3vl_2b_instr_test.sh <MODALITY> [GPU_ID]
# e.g.:
#   bash eval_qwen3vl_2b_instr_test.sh MRI
#   bash eval_qwen3vl_2b_instr_test.sh MRI 0
#   bash eval_qwen3vl_2b_instr_test.sh CT 1

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <MODALITY> [GPU_ID]"
    echo "Available MODALITIES: MRI, CT, XRay, Ultrasound, Dermoscopy, Fundus, OCT, Microscopy"
    echo "GPU_ID: Optional, specify which GPU to use (0-3). Default: auto"
    exit 1
fi

SELECTED_MODALITY="$1"
GPU_ID="${2:-}"  # Optional GPU ID, empty means auto

# Model configuration
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"  # Can use HF Hub path or local path
MODEL_NAME="Qwen3VL_2B_Instr"
BATCH_SIZE=16  # Reduced from 32 to prevent OOM
MODE="nothink"  # "nothink" or "think"

# Paths
BASE_OUTPUT_DIR="/mnt/task_runtime/results/${MODEL_NAME}"
EVAL_JSON_DIR="/mnt/task_runtime/data/omni_med_vqa_processed_sampled/eval_json"
IMAGE_FOLDER="/mnt/task_runtime/data//OmniMedVQA/OmniMedVQA"

# Modality mapping: (modality_name, json_filename)
declare -A MODALITY_TO_JSON
MODALITY_TO_JSON["MRI"]="MRI_test_300.json"
MODALITY_TO_JSON["CT"]="CT_test_300.json"
MODALITY_TO_JSON["XRay"]="X-Ray_test_300.json"
MODALITY_TO_JSON["Ultrasound"]="Ultrasound_test_300.json"
MODALITY_TO_JSON["Dermoscopy"]="Dermoscopy_test_300.json"
MODALITY_TO_JSON["Fundus"]="Fundus_test_300.json"
MODALITY_TO_JSON["OCT"]="OCT_test_300.json"
MODALITY_TO_JSON["Microscopy"]="Microscopy_test_300.json"

if [[ -z "${MODALITY_TO_JSON[${SELECTED_MODALITY}]}" ]]; then
    echo "ERROR: Invalid modality: ${SELECTED_MODALITY}"
    echo "Valid modalities: MRI, CT, XRay, Ultrasound, Dermoscopy, Fundus, OCT, Microscopy"
    exit 2
fi

# Create output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Choose evaluation script based on mode
if [ "${MODE}" == "think" ]; then
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa.py"
else
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa_nothink.py"
fi

echo "=========================================="
echo "Evaluating ${MODEL_NAME} (${MODE} mode)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${BASE_OUTPUT_DIR}"
echo "Modality: ${SELECTED_MODALITY}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU: ${GPU_ID}"
else
    echo "GPU: auto"
fi
echo "=========================================="

modality_name="${SELECTED_MODALITY}"
json_filename="${MODALITY_TO_JSON[${SELECTED_MODALITY}]}"

PROMPT_PATH="${EVAL_JSON_DIR}/${json_filename}"
OUTPUT_PATH="${BASE_OUTPUT_DIR}/test_${modality_name}_${MODEL_NAME}_results.json"

echo ""
echo "----------------------------------------"
echo "Processing: ${modality_name}"
echo "Prompt: ${PROMPT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "----------------------------------------"

# Check if prompt file exists
if [ ! -f "${PROMPT_PATH}" ]; then
    echo "ERROR: Prompt file not found: ${PROMPT_PATH}"
    exit 3
fi

# Clear GPU cache before each modality (use specified GPU if provided)
if [ -n "${GPU_ID}" ]; then
    python -c "import torch; torch.cuda.set_device(${GPU_ID}); torch.cuda.empty_cache()" 2>/dev/null || true
else
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

# Run evaluation with retry mechanism for OOM
RETRY_COUNT=0
MAX_RETRIES=2
CURRENT_BSZ="${BATCH_SIZE}"
SUCCESS=false

while [ $RETRY_COUNT -le $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        # Reduce batch size on retry
        CURRENT_BSZ=$((CURRENT_BSZ / 2))
        if [ $CURRENT_BSZ -lt 1 ]; then
            CURRENT_BSZ=1
        fi
        echo "   ⚠ Retrying with reduced batch_size=${CURRENT_BSZ}..."
        if [ -n "${GPU_ID}" ]; then
            python -c "import torch; torch.cuda.set_device(${GPU_ID}); torch.cuda.empty_cache()" 2>/dev/null || true
        else
            python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        fi
        sleep 2
    fi
    
    # Run evaluation and capture output
    GPU_ARGS=""
    if [ -n "${GPU_ID}" ]; then
        GPU_ARGS="--gpu_id ${GPU_ID}"
    fi
    
    if python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --batch_size "${CURRENT_BSZ}" \
        --output_path "${OUTPUT_PATH}" \
        --prompt_path "${PROMPT_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        ${GPU_ARGS} 2>&1 | tee /tmp/eval_${modality_name}.log; then
        echo "✓ ${modality_name} evaluation completed (batch_size=${CURRENT_BSZ})"
        SUCCESS=true
    else
        EXIT_CODE=$?
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        # Check if it's an OOM error
        if grep -qi "out of memory\|CUDA out of memory" /tmp/eval_${modality_name}.log 2>/dev/null; then
            if [ $RETRY_COUNT -le $MAX_RETRIES ]; then
                echo "   ⚠ OOM detected, will retry with smaller batch size..."
            else
                echo "✗ ${modality_name} evaluation failed: OOM after ${MAX_RETRIES} retries"
            fi
        else
            echo "✗ ${modality_name} evaluation failed (exit code: ${EXIT_CODE})"
            SUCCESS=false
            break
        fi
    fi
done

# Clear GPU cache after evaluation (use specified GPU if provided)
if [ -n "${GPU_ID}" ]; then
    python -c "import torch; torch.cuda.set_device(${GPU_ID}); torch.cuda.empty_cache()" 2>/dev/null || true
else
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi
sleep 1

echo ""
echo "=========================================="
echo "Evaluation completed for modality: ${SELECTED_MODALITY}"
echo "Results saved to: ${OUTPUT_PATH}"
echo "=========================================="

