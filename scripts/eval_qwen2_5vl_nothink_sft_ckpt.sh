#!/bin/bash

PYTHONPATH=. python src/eval_vqa/test_qwen2_5vl_vqa_nothink.py \
    --model_path src/r1-v/output/SFT/SFT_5000 \
    --batch_size 32 \
    --output_path /home/fayang/output/Med-R1/Qwen2.5-VL-3B-Instruct-SFT-CT-test/results.json \
    --prompt_path /home/fayang/Med-R1/src/eval_vqa/prompts/modality/test/CT_test.json \
    --image_folder /data/datasets/OmniMedVQA/OmniMedVQA 