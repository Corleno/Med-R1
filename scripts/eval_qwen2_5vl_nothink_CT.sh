#!/bin/bash

python src/eval_vqa/test_qwen2_5vl_vqa_nothink.py \
    --model_path /home/fayang/output/Med-R1/training/GRPO/GPRO_CT_from_Qwen2_5VL \
    --batch_size 32 \
    --output_path /home/fayang/output/Med-R1/evaluation/modality/train_CT/test_CT_GRPO_Qwen2.5VL3B_results.json \
    --prompt_path "/data/datasets/OmniMedVQA/OmniMedVQA/eval_json/modality/CT(Computed_Tomography)_test.json" \
    --image_folder /data/datasets/OmniMedVQA/OmniMedVQA 