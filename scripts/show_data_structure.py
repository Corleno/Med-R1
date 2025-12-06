#!/usr/bin/env python3
"""
Script to visualize the data structure for presentation.
Run this to show your boss the data processing pipeline.
"""

import os
from pathlib import Path

def print_section(title, char="="):
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")

def check_dataset_exists(path):
    """Check if dataset exists and return info"""
    if os.path.exists(path):
        # Check if it's a valid HuggingFace dataset
        dataset_dict_json = os.path.join(path, "dataset_dict.json")
        if os.path.exists(dataset_dict_json):
            return "✓ Exists (HF Dataset)"
        return "✓ Exists (checking...)"
    return "✗ Not found"

def main():
    print_section("OmniMedVQA Data Processing Structure", "=")
    
    base_dir = "/data/datasets/OmniMedVQA/OmniMedVQA"
    
    # 1. Basic dataset (for SFT + GRPO)
    print_section("1. Basic Dataset (for SFT → GRPO pipeline)", "-")
    basic_path = os.path.join(base_dir, "open_access_sft_data_hf")
    print(f"Path: {basic_path}")
    print(f"Status: {check_dataset_exists(basic_path)}")
    print("\nUsage:")
    print("  - SFT training: configs/qwen2_5/qwen2_5vl_sft_config_*.yaml")
    print("    → dataset_name: /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf")
    print("  - GRPO training: scripts/grpo_script.sh")
    print("    → --dataset_name /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf")
    
    # 2. Cross-modality datasets
    print_section("2. Cross-Modality Datasets (8 modalities × 2 splits = 16 datasets)", "-")
    modalities = ["CT", "MRI", "X_Ray", "Ultrasound", "Dermoscopy", 
                  "Fundus", "OCT", "Microscopy"]
    
    print("Modality splits:")
    for modality in modalities:
        safe_modality = modality.replace("-", "_")
        train_path = os.path.join(base_dir, f"open_access_sft_data_hf_modality_{safe_modality}_train")
        test_path = os.path.join(base_dir, f"open_access_sft_data_hf_modality_{safe_modality}_test")
        
        train_status = check_dataset_exists(train_path)
        test_status = check_dataset_exists(test_path)
        
        print(f"\n  {modality}:")
        print(f"    Train: {train_path}")
        print(f"           {train_status}")
        print(f"    Test:  {test_path}")
        print(f"           {test_status}")
    
    # 3. Cross-task datasets
    print_section("3. Cross-Task Datasets (5 tasks × 2 splits = 10 datasets)", "-")
    tasks = [
        "Anatomy_Identification",
        "Disease_Diagnosis", 
        "Lesion_Grading",
        "Modality_Recognition",
        "Other_Biological_Attributes"
    ]
    
    print("Task splits:")
    for task in tasks:
        train_path = os.path.join(base_dir, f"open_access_sft_data_hf_task_{task}_train")
        test_path = os.path.join(base_dir, f"open_access_sft_data_hf_task_{task}_test")
        
        train_status = check_dataset_exists(train_path)
        test_status = check_dataset_exists(test_path)
        
        print(f"\n  {task.replace('_', ' ')}:")
        print(f"    Train: {train_path}")
        print(f"           {train_status}")
        print(f"    Test:  {test_path}")
        print(f"           {test_status}")
    
    # 4. Data reading locations
    print_section("4. Where Data is Loaded in Training Scripts", "-")
    
    print("SFT Training:")
    print("  Script: scripts/sft_vqa_2_5.sh")
    print("  Config: src/r1-v/configs/qwen2_5/qwen2_5vl_sft_config_*.yaml")
    print("  Config line 10: dataset_name: /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf")
    print("  Code: src/r1-v/src/open_r1/sft_2_5.py:212")
    print("        dataset = load_from_disk(script_args.dataset_name)")
    
    print("\nGRPO Training:")
    print("  Script: scripts/grpo_script.sh")
    print("  Script line 45: --dataset_name /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf")
    print("  Code: src/r1-v/src/open_r1/grpo_vqa_nothink.py")
    print("        (loads via load_from_disk)")
    
    # 5. Summary
    print_section("5. Summary", "-")
    print("Total datasets generated:")
    print("  - 1 basic dataset (for SFT + GRPO)")
    print("  - 16 modality-specific datasets (8 modalities × 2 splits)")
    print("  - 10 task-specific datasets (5 tasks × 2 splits)")
    print("  - Total: 27 datasets")
    print("\nAll datasets are in HuggingFace format and ready for training!")
    
    print_section("", "=")

if __name__ == "__main__":
    main()

