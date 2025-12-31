cd src/r1-v

export CUDA_VISIBLE_DEVICES=4,5
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"

# export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=8888 \
         -m src.open_r1.grpo_vqa_nothink \
         --output_dir /home/fayang/output/Med-R1/training/GRPO/GPRO_CT_from_Qwen2_5VL \
         --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
         --dataset_name "/mnt/task_runtime/data/omni_med_vqa_processed/open_access_sft_data_hf_modality_CT(Computed_Tomography)_train" \
         --deepspeed local_scripts/zero3.json \
         --max_prompt_length 1024 \
         --max_completion_length 1024 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 2 \
         --logging_steps 1 \
         --bf16 true \
         --report_to wandb \
         --gradient_checkpointing true \
         --attn_implementation flash_attention_2 \
         --max_pixels 401408 \
         --num_train_epochs 1 \
         --run_name Qwen2.5-VL-3B-GRPO-CT \
         --save_steps  100 \
         --save_only_model true \
         --num_generations 8