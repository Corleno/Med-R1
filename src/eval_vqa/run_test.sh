#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.out
#SBATCH --error=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.err

# 加载环境
source ~/.bashrc
conda activate r1-v

export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 检查 GPU
echo "Checking GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 缓存目录
export TRANSFORMERS_CACHE=/local/scratch/ylai76/cache
export TRITON_CACHE=/local/scratch/ylai76/triton_cache
export HF_HOME=/local/scratch/ylai76/huggingface_cache

echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TRITON_CACHE: $TRITON_CACHE"
echo "HF_HOME: $HF_HOME"


# 进入代码目录
cd /local/scratch/ylai76/Code/R1-V/src/eval_vqa

# 获取 `sbatch` 传递的参数
MODEL_PATH="$1"
PROMPT_PATH="$2"
BSZ=64
OUTPUT_DIR="$3"

# 确保参数不为空
if [ -z "$MODEL_PATH" ] || [ -z "$PROMPT_PATH" ]; then
    echo "Error: Missing model path or JSON file!"
    exit 1
fi

# 生成输出路径
FILE_NAME=$(basename -- "$PROMPT_PATH")

OUTPUT_PATH="$OUTPUT_DIR/${FILE_NAME%.json}_result.json"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "MODEL_PATH: $MODEL_PATH"
echo "PROMPT_PATH: $PROMPT_PATH"
echo "FILE_NAME: $FILE_NAME"
echo "OUTPUT_PATH: $OUTPUT_PATH"

# 运行 Python 脚本
python test_qwen2_5vl_vqa.py --model_path "$MODEL_PATH" \
                           --batch_size "$BSZ" \
                           --output_path "$OUTPUT_PATH" \
                           --prompt_path "$PROMPT_PATH"

echo "Finished processing $PROMPT_PATH"