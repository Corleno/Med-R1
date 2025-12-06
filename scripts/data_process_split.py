"""
Independent data processing script: 80/20 split by modality and task type
for cross-modality and cross-task experiments.

This script is completely independent of data_process.py and will not
affect the original data processing pipeline.
"""

import json
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image as PILImage
from datasets import DatasetDict, Dataset, Features, Value, Image as DatasetImage

# ============================================================================
# Configuration paths (consistent with data_process.py)
# ============================================================================

open_access_dir = "/data/datasets/OmniMedVQA/OmniMedVQA/QA_information/Open-access"
dataset_root = "/data/datasets/OmniMedVQA/OmniMedVQA"

# ============================================================================
# Reuse functions from data_process.py (copied here to avoid dependencies)
# ============================================================================

def convert_raw_data_to_sft_data(data):
    """
    Convert raw data to SFT data
    Copied from data_process.py to maintain format consistency

    data example:
    {
        'dataset': 'ACRIMA',
        'question_id': 'ACRIMA_0000',
        'question_type': 'Modality Recognition',
        'question': 'What imaging technique was employed to obtain this picture?',
        'gt_answer': 'Fundus imaging',
        'image_path': 'Images/ACRIMA/Im553_g_ACRIMA.png',
        'option_A': 'PET scan',
        'option_B': 'CT scan',
        'option_C': 'Blood test',
        'option_D': 'Fundus imaging',
        'modality_type': 'Fundus Photography'
    }

    SFT data example:
    {
        "image": "path/to/image.jpg",
        "problem": "question",
        "solution": "answer"
    }
    """
    sft_data = []
    for item in data:
        required_keys = ["image_path", "question", "gt_answer"]
        option_keys = [k for k in item.keys() if k.startswith("option_")]
        assert len(option_keys) > 0, f"No option keys found in item: {item}"
        required_keys.extend([k for k in option_keys if k not in required_keys])

        # Build the multiple-choice prompt
        options_str = "\n".join([f"{k[-1]}: {item[k]}" for k in option_keys])

        # Map the ground-truth answer text back to its option letter
        gt_text = str(item["gt_answer"]).strip()
        correct_letter = None
        for k in option_keys:
            opt_text = str(item[k]).strip()
            if opt_text.lower().rstrip(".") == gt_text.lower().rstrip("."):
                correct_letter = k[-1]
                break

        if correct_letter is None:
            correct_token = gt_text
        else:
            correct_token = correct_letter

        sft_data.append(
            {
                "image": item["image_path"],
                "problem": item["question"] + "\n" + options_str,
                "solution": f"<answer> {correct_token} </answer>",
            }
        )

    return sft_data


@contextmanager
def pushd(path):
    """Context manager for changing directory"""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_image_from_path(image_path, dataset_root=None):
    """Load and resize image to 384x384"""
    try:
        # Convert relative path to absolute path if needed
        if not os.path.isabs(image_path):
            if dataset_root is None:
                dataset_root = "/data/datasets/OmniMedVQA/OmniMedVQA"
            image_path = os.path.join(dataset_root, image_path)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        img = PILImage.open(image_path).convert("RGB")
        img = img.resize((384, 384))
        return img
    except Exception as e:
        # Only print error for debugging, don't spam output
        # print(f"Error loading image {image_path}: {str(e)}")
        return None


# ============================================================================
# Split-related functions
# ============================================================================

def normalize_modality_name(modality_str):
    """
    Normalize modality_type from raw data to the 8 modalities in the paper.
    
    The 8 modalities in the paper:
    - CT (15,808 samples)
    - MRI (31,877 samples)
    - X-Ray (7,916 samples)
    - Ultrasound (10,991 samples)
    - Dermoscopy (6,679 samples)
    - Fundus (5,398 samples)
    - OCT (4,646 samples)
    - Microscopy (5,680 samples)
    """
    modality_str = str(modality_str).strip()
    # Mapping rules (adjust based on actual data)
    modality_map = {
        # CT
        'CT': 'CT',
        'Computed Tomography': 'CT',
        # MRI
        'MRI': 'MRI',
        'Magnetic Resonance Imaging': 'MRI',
        # X-Ray
        'X-Ray': 'X-Ray',
        'X-ray': 'X-Ray',
        'X-Ray Imaging': 'X-Ray',
        # Ultrasound
        'Ultrasound': 'Ultrasound',
        'US': 'Ultrasound',
        # Dermoscopy
        'Dermoscopy': 'Dermoscopy',
        'Dermatoscopy': 'Dermoscopy',
        # Fundus
        'Fundus Photography': 'Fundus',
        'Fundus': 'Fundus',
        'FP': 'Fundus',
        # OCT
        'OCT': 'OCT',
        'Optical Coherence Tomography': 'OCT',
        # Microscopy
        'Microscopy': 'Microscopy',
        'Micro': 'Microscopy',
    }
    # Try exact match
    if modality_str in modality_map:
        return modality_map[modality_str]
    # Try case-insensitive match
    for key, value in modality_map.items():
        if key.lower() == modality_str.lower():
            return value
    # If no match, return original value (can be manually checked later)
    return modality_str


def normalize_task_name(task_str):
    """
    Normalize question_type from raw data to the 5 task types in the paper.
    
    The 5 task types in the paper:
    - Anatomy Identification (16,448 samples)
    - Disease Diagnosis (55,387 samples)
    - Lesion Grading (2,098 samples)
    - Modality Recognition (11,565 samples)
    - Other Biological Attributes (3,498 samples)
    """
    task_str = str(task_str).strip()
    task_map = {
        'Anatomy Identification': 'Anatomy Identification',
        'Disease Diagnosis': 'Disease Diagnosis',
        'Lesion Grading': 'Lesion Grading',
        'Modality Recognition': 'Modality Recognition',
        'Other Biological Attributes': 'Other Biological Attributes',
        'Biological Attribute Analysis': 'Other Biological Attributes',
    }
    if task_str in task_map:
        return task_map[task_str]
    for key, value in task_map.items():
        if key.lower() == task_str.lower():
            return value
    return task_str


def split_data_by_group(data, group_key_func, train_ratio=0.8, seed=42):
    """
    Group data by group_key_func, then perform train/test split for each group.
    
    Args:
        data: List of raw data items
        group_key_func: Function that takes an item and returns a group key
        train_ratio: Training ratio (default 0.8, i.e., 80/20 split)
        seed: Random seed
    Returns:
        dict: {group_key: {'train': [...], 'test': [...]}}
    """
    random.seed(seed)
    
    # Group by group_key
    grouped_data = defaultdict(list)
    for item in data:
        group_key = group_key_func(item)
        grouped_data[group_key].append(item)
    
    # Perform split for each group
    splits = {}
    for group_key, items in grouped_data.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        splits[group_key] = {
            'train': items[:split_idx],
            'test': items[split_idx:]
        }
        print(f"  {group_key}: {len(items)} samples -> "
              f"train: {len(splits[group_key]['train'])}, "
              f"test: {len(splits[group_key]['test'])}")
    
    return splits


def process_single_split(group_key, split_name, raw_items, split_type, 
                         max_samples_per_split, num_workers, dataset_root,
                         save_json=False):
    """
    Process a single split (e.g., CT_train) and save it.
    This function is designed to be called in parallel.
    """
    # Define the features for the HuggingFace dataset
    features = Features({
        "image": DatasetImage(),
        "problem": Value("string"),
        "solution": Value("string"),
    })
    
    # Clean special characters in group_key for file paths
    safe_key = group_key.replace(' ', '_').replace('/', '_')
    
    # Limit sample count (for testing)
    if max_samples_per_split is not None and len(raw_items) > max_samples_per_split:
        print(f"    [{group_key} {split_name}] Limiting to "
              f"{max_samples_per_split} samples (for testing)")
        raw_items = raw_items[:max_samples_per_split]
    
    # Convert to SFT format
    sft_items = convert_raw_data_to_sft_data(raw_items)
    
    # Build HuggingFace dataset
    hf_dict = {
        "image": [],
        "problem": [],
        "solution": [],
    }
    
    # Also save image paths for JSON conversion (only if save_json is True)
    image_paths = []
    
    # Remove pushd to avoid parallel chdir conflicts
    # Use absolute paths throughout
    print(f"    [{group_key} {split_name}] Processing "
          f"{len(sft_items)} samples (using {num_workers} threads)...")
    
    # Parallel image loading using ThreadPoolExecutor
    def load_image_with_metadata(item):
        """Load image and return with metadata"""
        # Pass dataset_root to ensure correct path resolution
        img = load_image_from_path(item["image"], dataset_root=dataset_root)
        return {
            "image": img,
            "problem": item["problem"],
            "solution": item["solution"],
            "success": img is not None
        }
    
    # Use ThreadPoolExecutor for parallel image loading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks, keeping track of which item corresponds to which future
        future_to_item = {
            executor.submit(load_image_with_metadata, item): item
            for item in sft_items
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_item), 
                          total=len(sft_items), 
                          desc=f"  {group_key} {split_name}"):
            result = future.result()
            if result["success"]:  # Skip failed image loads
                hf_dict["image"].append(result["image"])
                hf_dict["problem"].append(result["problem"])
                hf_dict["solution"].append(result["solution"])
                # Save original image path for JSON conversion (if needed)
                if save_json:
                    original_item = future_to_item[future]
                    image_paths.append(original_item["image"])
    
    # If no valid images were loaded, skip this split to avoid empty datasets
    num_examples = len(hf_dict["image"])
    if num_examples == 0:
        print(f"    [{group_key} {split_name}] No valid images loaded, skipping this split.")
    else:
        print(f"    [{group_key} {split_name}] Building DatasetDict with {num_examples} examples...")
        # Create DatasetDict
        split_dataset = Dataset.from_dict(hf_dict, features=features)
        dataset_dict = DatasetDict({split_name: split_dataset})
        
        # Save path: open_access_sft_data_hf_modality_CT_train, etc.
        save_path_split = (f"open_access_sft_data_hf_{split_type}_"
                         f"{safe_key}_{split_name}")
        
        # Use absolute path from the start
        abs_save_path = os.path.join(dataset_root, save_path_split)
        
        # Check if HF dataset already exists
        if os.path.exists(abs_save_path):
            print(f"    [{group_key} {split_name}] HF dataset already exists: {save_path_split}")
            print(f"    [{group_key} {split_name}] Skipping HF save (delete to regenerate)")
        else:
            print(f"    [{group_key} {split_name}] "
                  f"Saving to: {abs_save_path}")
            try:
                # Use absolute path directly, no pushd needed
                dataset_dict.save_to_disk(abs_save_path)
                # Verify the save was successful
                if not os.path.exists(abs_save_path):
                    raise RuntimeError(
                        f"Save completed but folder not found: "
                        f"{abs_save_path}")
                print(f"    [{group_key} {split_name}] "
                      f"Successfully saved to: {save_path_split}")
            except Exception as e:
                print(f"    ✗ [{group_key} {split_name}] "
                      f"ERROR saving HF dataset: {e}")
                print(f"    [{group_key} {split_name}] "
                      f"Save path: {save_path_split}")
                print(f"    [{group_key} {split_name}] "
                      f"Full path: {abs_save_path}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to be caught by outer handler
            
            # Also save JSON file for evaluation (only for test splits and if save_json is True)
            if split_name == "test" and save_json:
                # Check if JSON file already exists
                json_dir = os.path.join(dataset_root, "eval_json", split_type)
                os.makedirs(json_dir, exist_ok=True)
                json_path = os.path.join(json_dir, f"{safe_key}_test.json")
                
                if os.path.exists(json_path):
                    print(f"    [{group_key} {split_name}] JSON file already exists: {json_path}")
                    print(f"    [{group_key} {split_name}] Skipping JSON generation (delete to regenerate)")
                else:
                    json_data = [
                        {
                            "image": img_path,
                            "problem": prob,
                            "solution": sol
                        }
                        for img_path, prob, sol in zip(
                            image_paths, hf_dict["problem"], hf_dict["solution"]
                        )
                    ]
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    print(f"    [{group_key} {split_name}] Saved JSON to: {json_path}")
            
            print(f"    [{group_key} {split_name}] Completed!")
    
    return f"{group_key} {split_name}"


def save_split_dataset(splits, split_type, max_samples_per_split=None, 
                      num_workers=4, parallel_datasets=4, 
                      test_only=False, save_json=False):
    """
    Convert splits to SFT format and save as HuggingFace datasets.
    Now processes multiple datasets in parallel.
    
    Args:
        splits: {group_key: {'train': [...], 'test': [...]}}
        split_type: 'modality' or 'task'
        max_samples_per_split: Maximum samples per split (for testing,
            None means no limit)
        num_workers: Number of threads for parallel image loading within each dataset
        parallel_datasets: Number of datasets to process in parallel (default: 4)
    """
    # Collect all (group_key, split_name) pairs to process
    tasks = []
    for group_key, split_data in splits.items():
        if test_only:
            # Only process test splits
            if 'test' in split_data:
                tasks.append((group_key, 'test', split_data['test']))
        else:
            # Process both train and test splits
            for split_name in ['train', 'test']:
                tasks.append((group_key, split_name, split_data[split_name]))
    
    if test_only:
        print(f"    Processing {len(tasks)} test splits only (skipping train splits)")
    else:
        print(f"    Total {len(tasks)} splits to process")
    print(f"    Processing {parallel_datasets} datasets in parallel")
    print(f"    Each dataset uses {num_workers} threads for image loading")
    
    # Process datasets in parallel
    with ThreadPoolExecutor(max_workers=parallel_datasets) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_split,
                group_key, split_name, raw_items, split_type,
                max_samples_per_split, num_workers, dataset_root,
                save_json=save_json
            ): (group_key, split_name)
            for group_key, split_name, raw_items in tasks
        }
        
        # Wait for all tasks to complete
        for future in tqdm(as_completed(futures), 
                          total=len(futures), 
                          desc="Processing datasets"):
            group_key, split_name = futures[future]
            try:
                result = future.result()
                print(f"    ✓ Completed: {result}")
            except Exception as e:
                print(f"    ✗ Error processing {group_key} {split_name}: {e}")
                raise


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate 80/20 split datasets by modality and task type"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only process test splits (skip train splits)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save JSON files for test splits (for evaluation scripts)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Generating 80/20 split datasets by modality and task type")
    if args.test_only:
        print("Mode: Processing test splits only")
    if args.save_json:
        print("Mode: Will save JSON files for test splits")
    print("="*80)
    
    # 1. Load raw data
    print("\n[Step 1/4] Loading raw data...")
    open_access_files = os.listdir(open_access_dir)
    print(f"Found {len(open_access_files)} JSON files")
    
    open_access_data = []
    for file in open_access_files:
        with open(os.path.join(open_access_dir, file), "r") as f:
            data = json.load(f)
            open_access_data.extend(data)
    
    print(f"Total loaded {len(open_access_data)} raw data items")
    
    # 2. Group by modality and split
    print("\n[Step 2/4] Grouping by modality_type and performing 80/20 split...")
    modality_splits = split_data_by_group(
        open_access_data,
        group_key_func=lambda item: normalize_modality_name(
            item.get('modality_type', 'Unknown')
        )
    )
    
    # 3. Group by question_type and split
    print("\n[Step 3/4] Grouping by question_type and performing 80/20 split...")
    task_splits = split_data_by_group(
        open_access_data,
        group_key_func=lambda item: normalize_task_name(
            item.get('question_type', 'Unknown')
        )
    )
    
    # 4. Convert to SFT format and save
    print("\n[Step 4/4] Converting to SFT format and saving...")
    
    # Save modality splits
    print("\nSaving modality splits...")
    # For testing, set max_samples_per_split to a small number first
    # After confirming it works, set to None to process all data
    # num_workers: threads per dataset for image loading
    # parallel_datasets: number of datasets to process simultaneously
    save_split_dataset(modality_splits, 'modality', 
                      max_samples_per_split=None,
                      num_workers=8,      # Threads per dataset for image loading
                      parallel_datasets=18,  # Number of datasets in parallel
                      test_only=args.test_only,
                      save_json=args.save_json)
    
    # Save task splits
    print("\nSaving task splits...")
    save_split_dataset(task_splits, 'task', 
                      max_samples_per_split=None,
                      num_workers=8,      # Threads per dataset for image loading
                      parallel_datasets=18,  # Number of datasets in parallel
                      test_only=args.test_only,
                      save_json=args.save_json)
    
    print("\n" + "="*80)
    print("Split dataset generation completed!")
    print("="*80)
    print("\nGenerated dataset path format:")
    print("  - Cross-modality: "
          "open_access_sft_data_hf_modality_{MODALITY}_{train|test}")
    print("  - Cross-task: "
          "open_access_sft_data_hf_task_{TASK}_{train|test}")
    print("\nExamples:")
    print("  - open_access_sft_data_hf_modality_CT_train")
    print("  - open_access_sft_data_hf_modality_CT_test")
    print("  - open_access_sft_data_hf_task_Disease_Diagnosis_train")
    print("  - open_access_sft_data_hf_task_Disease_Diagnosis_test")


if __name__ == "__main__":
    main()

