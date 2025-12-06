#!/bin/bash

python src/eval_vqa/test_qwen2_5vl_vqa_nothink.py \
    --model_path /home/fayang/checkpoints/Qwen2.5-VL-3B-Instruct \
    --batch_size 32 \
    --output_path /home/fayang/output/Med-R1/Qwen2.5-VL-3B-Instruct-CT-test/results.json \
    --prompt_path /home/fayang/Med-R1/src/eval_vqa/prompts/modality/test/CT_test.json \
    --image_folder /data/datasets/OmniMedVQA/OmniMedVQA 