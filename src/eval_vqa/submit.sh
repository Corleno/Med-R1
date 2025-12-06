#!/bin/bash

# 目录配置
MODEL_PATH="/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT"  # 模型路径
TEST_DIR="/local/scratch/ylai76/Code/R1-V/src/eval_vqa/prompts/modality/test"
LOG_DIR="/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/Qwen2_5_3B/Modality_think/VQA_CT" 

# 遍历所有 JSON 文件
for PROMPT_PATH in "$TEST_DIR"/*.json; do
    # 提取文件名（不带路径）
    FILE_NAME=$(basename -- "$PROMPT_PATH")

    # 生成 SLURM 任务名称
    JOB_NAME="eval_${FILE_NAME%.json}"

    echo "Submitting job for $PROMPT_PATH with model $MODEL_PATH..."
    
    # 提交 sbatch 任务，传递 `MODEL_PATH` 和 `PROMPT_PATH`
    sbatch --job-name="$JOB_NAME" run_test.sh "$MODEL_PATH" "$PROMPT_PATH" "$LOG_DIR"

    echo "Job $JOB_NAME submitted!"
done

echo "All jobs submitted!"