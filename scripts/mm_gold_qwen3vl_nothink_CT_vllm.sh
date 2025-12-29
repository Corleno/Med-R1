#!/bin/bash
cd /mnt/task_runtime/Med-R1

# 1) Kill any existing vLLM / torchrun processes
pkill -f "vllm serve" || true
pkill -f "trl vllm-serve" || true
pkill -f "src.gold_multimodal.gold_multimodal" || pkill -f torchrun || true

teacher_model_name_or_path=Qwen/Qwen3-VL-4B-Instruct
student_model_name_or_path=Qwen/Qwen3-VL-2B-Instruct
vllm_server_port=8001
vllm_server_host=127.0.0.1

# Start vllm server
if ! pgrep -f "trl vllm-serve --model $teacher_model_name_or_path" > /dev/null; then
    echo "Starting vllm server..."
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model $teacher_model_name_or_path --port $vllm_server_port --host $vllm_server_host --dtype bfloat16 --trust_remote_code &
else
    echo "trl vllm-server is already running."
fi

while ! curl -s http://$vllm_server_host:$vllm_server_port/health/ > /dev/null; do
  echo "Waiting for TRL vllm server..."
  sleep 10
done

echo "trl vllm-server is ready."

# Multimodal Gold training script for Qwen3-VL

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

torchrun --nproc_per_node=7 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port=12346 \
         -m src.gold_multimodal.gold_multimodal \
         --model_name_or_path $student_model_name_or_path \
         --deepspeed src/training_configs/zero3.json \
         --teacher_model_name_or_path $teacher_model_name_or_path \
         --dataset_name "/mnt/task_runtime/data/omni_med_vqa_processed/open_access_sft_data_hf_modality_CT(Computed_Tomography)_train" \
         --learning_rate 2e-5 \
         --logging_steps 1 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 8 \
         --output_dir gold-model \
         --num_train_epochs 1 \
         --use_vllm True \
         --vllm_server_port $vllm_server_port \
         --vllm_server_host $vllm_server_host