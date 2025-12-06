#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.out
#SBATCH --error=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.err

source ~/.bashrc
conda activate r1-v

export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# GPU 检查
echo "Checking GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

export TRANSFORMERS_CACHE=/local/scratch/ylai76/cache
export TRITON_CACHE=/local/scratch/ylai76/triton_cache
export HF_HOME=/local/scratch/ylai76/huggingface_cache

echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TRITON_CACHE: $TRITON_CACHE"
echo "HF_HOME: $HF_HOME"

cd /local/scratch/ylai76/Code/R1-V/src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"

MASTER_PORT=$((12000 + RANDOM % 2000))
echo "Using master port: $MASTER_PORT"

# 修改点：增加 --deepspeed 参数，指定 DeepSpeed 配置文件路径
torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_vqa_nothink.py \
    --deepspeed /local/scratch/ylai76/Code/R1-V/src/r1-v/configs/ds_config.json \
    --output_dir /local/scratch/ylai76/Code/R1-V/output/Qwen2_7B_modality_nothink/VQA_CT \
    --model_name_or_path /local/scratch/ylai76/Code/R1-V/model/Qwen2-VL-7B-Instruct/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac \
    --dataset_name /local/scratch/ylai76/Code/R1-V/data/VQA/384/CT \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Deepspeed_Qwen2-VL-7B-GRPO-nothink-CT \
    --save_steps 2000 \
    --save_only_model true \
    --num_generations 4