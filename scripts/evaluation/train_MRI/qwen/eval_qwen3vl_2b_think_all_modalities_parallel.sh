#!/bin/bash

# Evaluate Qwen3-VL-2B (Thinking mode) on all 8 modalities using 4 GPUs in parallel
# Training: MRI (600 samples)
# Testing: 8 modalities × 300 samples each
# Each GPU processes 2 modalities

# Model configuration
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME="Qwen3VL_2B_Think"
BATCH_SIZE=16  # Reduced to prevent OOM
MODE="think"

# Paths
BASE_OUTPUT_DIR="/home/fayang/output/Med-R1/evaluation/modality/train_MRI/qwen/${MODEL_NAME}"
EVAL_JSON_DIR="/data/datasets/OmniMedVQA/radiology_sampled/eval_json"
IMAGE_FOLDER="/data/datasets/OmniMedVQA/OmniMedVQA"

# Choose evaluation script based on mode
if [ "${MODE}" == "think" ]; then
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa.py"
else
    EVAL_SCRIPT="src/eval_vqa/test_qwen3vl_vqa_nothink.py"
fi

# Create output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Modality mapping: (modality_name, json_filename, gpu_id)
# Distribute 8 modalities across 4 GPUs (2 per GPU)
# Optimized allocation: Mix large and small modalities for better load balancing
# Based on image size analysis: Fundus (0.80MB) and Dermoscopy (0.41MB) are largest
declare -a MODALITIES=(
    "Fundus:Fundus_test_300.json:0"      # Largest + Smallest
    "Ultrasound:Ultrasound_test_300.json:0"
    "Dermoscopy:Dermoscopy_test_300.json:1"  # Large + Small
    "OCT:OCT_test_300.json:1"
    "Microscopy:Microscopy_test_300.json:2"  # Medium + Small
    "CT:CT_test_300.json:2"
    "XRay:X-Ray_test_300.json:3"          # Medium + Small
    "MRI:MRI_test_300.json:3"
)

echo "=========================================="
echo "Evaluating ${MODEL_NAME} (${MODE} mode)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${BASE_OUTPUT_DIR}"
echo "Using 4 GPUs in parallel"
echo "=========================================="

# Function to check if result file exists and is complete
check_result_complete() {
    local output_path=$1
    local expected_samples=300
    
    if [ ! -f "${output_path}" ]; then
        return 1  # File doesn't exist
    fi
    
    # Check if file is valid JSON and has expected number of samples
    local sample_count=$(python3 -c "
import json
try:
    with open('${output_path}', 'r') as f:
        data = json.load(f)
        results = data.get('results', [])
        print(len(results))
except:
    print(0)
" 2>/dev/null)
    
    if [ "${sample_count}" = "${expected_samples}" ]; then
        return 0  # Complete
    else
        return 1  # Incomplete or invalid
    fi
}

# Function to evaluate a single modality on a specific GPU
evaluate_modality() {
    local modality_name=$1
    local json_filename=$2
    local gpu_id=$3
    
    local PROMPT_PATH="${EVAL_JSON_DIR}/${json_filename}"
    local OUTPUT_PATH="${BASE_OUTPUT_DIR}/test_${modality_name}_${MODEL_NAME}_results.json"
    local LOG_FILE="/tmp/eval_${modality_name}_gpu${gpu_id}.log"
    
    # Check if result already exists and is complete
    if check_result_complete "${OUTPUT_PATH}"; then
        echo "[GPU ${gpu_id}] ⏭️  Skipping ${modality_name} - result already exists and is complete"
        return 0
    fi
    
    echo "[GPU ${gpu_id}] Starting ${modality_name} evaluation..."
    
    # Clear GPU cache before evaluation (use the actual GPU ID)
    python -c "import torch; torch.cuda.set_device(${gpu_id}); torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Run evaluation with retry mechanism for OOM
    local RETRY_COUNT=0
    local MAX_RETRIES=2
    local CURRENT_BSZ="${BATCH_SIZE}"
    local SUCCESS=false
    
    while [ $RETRY_COUNT -le $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
        if [ $RETRY_COUNT -gt 0 ]; then
            CURRENT_BSZ=$((CURRENT_BSZ / 2))
            if [ $CURRENT_BSZ -lt 1 ]; then
                CURRENT_BSZ=1
            fi
            echo "[GPU ${gpu_id}] ⚠ Retrying ${modality_name} with batch_size=${CURRENT_BSZ}..."
            python -c "import torch; torch.cuda.set_device(${gpu_id}); torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 2
        fi
        
        # Run evaluation (don't use CUDA_VISIBLE_DEVICES, use --gpu_id instead)
        if python "${EVAL_SCRIPT}" \
            --model_path "${MODEL_PATH}" \
            --batch_size "${CURRENT_BSZ}" \
            --output_path "${OUTPUT_PATH}" \
            --prompt_path "${PROMPT_PATH}" \
            --image_folder "${IMAGE_FOLDER}" \
            --gpu_id "${gpu_id}" \
            > "${LOG_FILE}" 2>&1; then
            echo "[GPU ${gpu_id}] ✓ ${modality_name} evaluation completed (batch_size=${CURRENT_BSZ})"
            SUCCESS=true
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            
            # Check if it's an OOM error
            if grep -qi "out of memory\|CUDA out of memory" "${LOG_FILE}" 2>/dev/null; then
                if [ $RETRY_COUNT -le $MAX_RETRIES ]; then
                    echo "[GPU ${gpu_id}] ⚠ ${modality_name} OOM detected, will retry with smaller batch size..."
                else
                    echo "[GPU ${gpu_id}] ✗ ${modality_name} evaluation failed: OOM after ${MAX_RETRIES} retries"
                fi
            else
                echo "[GPU ${gpu_id}] ✗ ${modality_name} evaluation failed"
                # Show last few lines of error
                echo "   Last error lines:"
                tail -5 "${LOG_FILE}" | sed 's/^/   /'
                SUCCESS=false
                break
            fi
        fi
    done
    
    # Clear GPU cache after evaluation
    python -c "import torch; torch.cuda.set_device(${gpu_id}); torch.cuda.empty_cache()" 2>/dev/null || true
    
    if [ "$SUCCESS" = true ]; then
        return 0
    else
        return 1
    fi
}

# Export function for parallel execution
export -f evaluate_modality
export -f check_result_complete
export MODEL_PATH MODEL_NAME BATCH_SIZE MODE BASE_OUTPUT_DIR EVAL_JSON_DIR IMAGE_FOLDER EVAL_SCRIPT

# Run evaluations: parallel across GPUs, serial within each GPU
# Group modalities by GPU to avoid memory conflicts
declare -A GPU_TASKS
for modality_entry in "${MODALITIES[@]}"; do
    IFS=':' read -r modality_name json_filename gpu_id <<< "${modality_entry}"
    GPU_TASKS["${gpu_id}"]="${GPU_TASKS["${gpu_id}"]}${modality_entry} "
done

echo ""
echo "=========================================="
echo "Starting evaluations"
echo "Tasks will run in parallel across GPUs"
echo "Tasks on the same GPU will run sequentially"
echo "=========================================="

# Function to run all tasks for a specific GPU (sequentially)
run_gpu_tasks() {
    local gpu_id=$1
    local tasks="${GPU_TASKS["${gpu_id}"]}"
    
    echo "[GPU ${gpu_id}] Starting tasks..."
    for modality_entry in $tasks; do
        IFS=':' read -r modality_name json_filename task_gpu_id <<< "${modality_entry}"
        
        # Verify GPU ID matches
        if [ "${task_gpu_id}" != "${gpu_id}" ]; then
            echo "ERROR: GPU ID mismatch for ${modality_name}"
            continue
        fi
        
        # Check if prompt file exists
        if [ ! -f "${EVAL_JSON_DIR}/${json_filename}" ]; then
            echo "ERROR: Prompt file not found: ${EVAL_JSON_DIR}/${json_filename}"
            continue
        fi
        
        # Run evaluation (sequentially on this GPU)
        evaluate_modality "${modality_name}" "${json_filename}" "${gpu_id}"
    done
    echo "[GPU ${gpu_id}] All tasks completed"
}

# Export function for parallel execution
export -f run_gpu_tasks
export -f evaluate_modality
export -f check_result_complete
export MODEL_PATH MODEL_NAME BATCH_SIZE MODE BASE_OUTPUT_DIR EVAL_JSON_DIR IMAGE_FOLDER EVAL_SCRIPT

# Run tasks for each GPU in parallel (background)
PIDS=()
for gpu_id in "${!GPU_TASKS[@]}"; do
    run_gpu_tasks "${gpu_id}" &
    PIDS+=($!)
    echo "Started GPU ${gpu_id} tasks (PID: ${PIDS[-1]})"
done

echo ""
echo "=========================================="
echo "All GPU task groups started in parallel"
echo "Waiting for completion..."
echo "=========================================="

# Wait for all background processes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "✓ GPU task group (PID: $PID) completed successfully"
    else
        echo "✗ GPU task group (PID: $PID) failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All evaluations completed successfully!"
else
    echo "⚠️  ${FAILED} evaluation(s) failed"
fi
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "=========================================="

