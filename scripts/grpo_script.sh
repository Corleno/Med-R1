cd src/r1-v

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"

# export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# export LOG_PATH="./debug_log_2b.txt"

# torchrun --nproc_per_node="8" \
#     --nnodes="1" \
#     --node_rank="0" \
#     --master_addr="127.0.0.1" \
#     --master_port="12345" \
#     src/open_r1/grpo.py \
#     --output_dir <OUTPUT_DIR> \
#     --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \ 
#     --dataset_name leonardPKU/clevr_cogen_a_train \  
#     --deepspeed local_scripts/zero3.json \
#     --max_prompt_length 512 \
#     --max_completion_length 512 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 1 \
#     --bf16 \
#     --report_to wandb \
#     --gradient_checkpointing false \
#     --attn_implementation flash_attention_2 \
#     --max_pixels 401408 \
#     --num_train_epochs 2 \
#     --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
#     --save_steps 100 \
#     --save_only_model true \
#     --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  


torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12345 \
         src/open_r1/grpo_vqa_nothink.py \
         --output_dir ./output/GPRO_5000_from_SFT_5000 \
         --model_name_or_path /home/fayang/Med-R1/src/r1-v/output/SFT/Qwen2.5-VL-SFT_5000 \
         --dataset_name /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf \
         --deepspeed local_scripts/zero3.json \
         --max_prompt_length 512 \
         --max_completion_length 512 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 1 \
         --logging_steps 1 \
         --bf16 \
         --report_to wandb \
         --gradient_checkpointing true \
         --attn_implementation flash_attention_2 \
         --max_pixels 401408 \
         --num_train_epochs 1 \
         --run_name Qwen2.5-VL-3B-GRPO-5000_from_SFT_5000 \
         --save_steps  100 \
         --save_only_model true \
         --num_generations 8