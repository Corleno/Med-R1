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
#!/bin/bash

echo "Checking GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

export TRANSFORMERS_CACHE=/local/scratch/ylai76/cache
export TRITON_CACHE=/local/scratch/ylai76/triton_cache
export HF_HOME=/local/scratch/ylai76/huggingface_cache

echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TRITON_CACHE: $TRITON_CACHE"
echo "HF_HOME: $HF_HOME"

cd /local/scratch/ylai76/Code/R1-V/src/eval_vqa

# 传递参数
MODEL_PATH="/local/scratch/ylai76/Code/R1-V/model/Qwen2-VL-2B-Instruct"
BSZ=64
OUTPUT_PATH="./logs/Chart_QA/qwen2_2B_ChartQA_nothink.json"
PROMPT_PATH="/local/scratch/ylai76/Code/R1-V/src/eval_vqa/prompts/ChartQA/sampled_train_2000.json"

# 运行 Python 代码并传入参数
python test_qwen2vl_vqa_zero_shot_ChartQA_nothink.py --model_path "$MODEL_PATH" --batch_size "$BSZ" --output_path "$OUTPUT_PATH" --prompt_path "$PROMPT_PATH"