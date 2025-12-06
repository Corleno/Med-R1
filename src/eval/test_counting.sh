#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
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

cd /local/scratch/ylai76/Code/R1-V/src/eval

torchrun /local/scratch/ylai76/Code/R1-V/src/eval/test_qwen2vl_counting_superclevr.py