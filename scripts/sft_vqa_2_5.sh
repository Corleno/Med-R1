#!/bin/bash

# Simple local SFT script for Qwen2.5-VL on OmniMedVQA VQA data.
# 注意：只在你自己的 scripts 目录下新建这个脚本，不改 repo 里任何模型代码。

set -e

# 可选：如果你是从项目根目录直接运行，可以在这里激活环境
# source ~/.bashrc
# conda activate med-r1

# 进入 r1-v 子项目目录
cd src/r1-v

# 和现在能跑通的 GRPO 脚本保持一致：用 4 张卡 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 为了尽量贴近作者的 run_sft_train_vqa_2_5.sh，我们改用 sft_2_5.py 和相应的配置文件。
# 不修改任何模型 / 训练脚本，只通过命令行参数和 config 适配你的路径。

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  src/open_r1/sft_2_5.py \
  --config configs/qwen2_5/qwen2_5vl_sft_config_5000.yaml


