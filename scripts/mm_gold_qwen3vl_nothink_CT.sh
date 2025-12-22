#!/bin/bash

# Multimodal Gold training script for Qwen3-VL

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.gold_multimodal.gold_multimodal \
         --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
         --deepspeed src/training_configs/zero3.json \
         --teacher_model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
         --dataset_name data/open_access_sft_data_hf_modality_OCT_train_test \
         --learning_rate 2e-5 \
         --logging_steps 1 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 8 \
         --output_dir gold-model \
         --num_train_epochs 1 \
         --push_to_hub \
         --gradient_checkpointing