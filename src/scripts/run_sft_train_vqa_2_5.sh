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

# GPU
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

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /local/scratch/ylai76/Code/R1-V/src/r1-v/configs/zero2.yaml /local/scratch/ylai76/Code/R1-V/src/r1-v/src/open_r1/sft_2_5.py --config /local/scratch/ylai76/Code/R1-V/src/r1-v/configs/qwen2_5/qwen2_5vl_sft_config_Der.yaml