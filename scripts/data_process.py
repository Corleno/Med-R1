import json

json_path = "/data/datasets/OmniMedVQA/OmniMedVQA/QA_information/Open-access/ACRIMA.json"
with open(json_path, "r") as f:
    acrima_data = json.load(f)

print(f"Loaded {len(acrima_data)} items from {json_path}")

restricted_json_path = "/data/datasets/OmniMedVQA/OmniMedVQA/QA_information/Restricted-access/AIDA.json"
with open(restricted_json_path, "r") as f:
    aida_data = json.load(f)

print(f"Loaded {len(aida_data)} items from {restricted_json_path}")

import os

open_access_dir = "/data/datasets/OmniMedVQA/OmniMedVQA/QA_information/Open-access"
open_access_files = os.listdir(open_access_dir)
print("Files in Open-access directory:", open_access_files)

restricted_access_dir = "/data/datasets/OmniMedVQA/OmniMedVQA/QA_information/Restricted-access"
restricted_access_files = os.listdir(restricted_access_dir)
print("Files in Restricted-access directory:", restricted_access_files)

# Combine all json files in the open-access directory
import json

open_access_data = []
for file in open_access_files:
    with open(os.path.join(open_access_dir, file), "r") as f:
        data = json.load(f)
        open_access_data.extend(data)

print(f"Total items in Open-access dataset: {len(open_access_data)}")

def convert_raw_data_to_sft_data(data):
    """
    Convert raw data to SFT data

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

    Args:
        data: list of raw data
    Returns:
        list of SFT data
    """
    sft_data = []
    for item in data:
        required_keys = ["image_path", "question", "gt_answer"]
        # Find all keys that start with "option_" in this item and add them to required_keys (if not already present)
        option_keys = [k for k in item.keys() if k.startswith("option_")]
        assert len(option_keys) > 0, f"No option keys found in item: {item}"
        required_keys.extend([k for k in option_keys if k not in required_keys])

        # Build the multiple-choice prompt: "Question\nA: xxx\nB: yyy\n..."
        options_str = "\n".join([f"{k[-1]}: {item[k]}" for k in option_keys])

        # Map the ground-truth answer text back to its option letter (A/B/C/...)
        gt_text = str(item["gt_answer"]).strip()
        correct_letter = None
        for k in option_keys:
            opt_text = str(item[k]).strip()
            if opt_text.lower().rstrip(".") == gt_text.lower().rstrip("."):
                correct_letter = k[-1]
                break

        # If we somehow cannot find a matching option, fall back to using the raw text
        if correct_letter is None:
            correct_token = gt_text
        else:
            correct_token = correct_letter

        sft_data.append(
            {
            "image": item["image_path"],
                "problem": item["question"] + "\n" + options_str,
                # For GRPO reward function we want the answer *letter*, e.g. "<answer> C </answer>"
                "solution": f"<answer> {correct_token} </answer>",
            }
        )

    return sft_data

# Convert open-access data to SFT data
open_access_sft_data = convert_raw_data_to_sft_data(open_access_data)

# Save the SFT data to a json file
with open("/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data.json", "w") as f:
    json.dump(open_access_sft_data, f)

# Load the SFT data from a json file
with open("/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data.json", "r") as f:
    open_access_sft_data = json.load(f)

import os
from datasets import DatasetDict, Dataset, Features, Value, Image as DatasetImage
from contextlib import contextmanager
from tqdm import tqdm
from PIL import Image as PILImage

save_path = "open_access_sft_data_hf"  # relative path; will be created in current working directory

@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

dataset_root = "/data/datasets/OmniMedVQA/OmniMedVQA"

hf_dict = {
    "image": [],
    "problem": [],
    "solution": [],
}

def load_image_from_path(image_path):
    try:
        img = PILImage.open(image_path).convert("RGB")
        # Resize to 384 x 384 as described in the README
        img = img.resize((384, 384))
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}. Image path: {image_path}")
        return None

# Define the features for the HuggingFace dataset, using PIL Image type for images
features = Features({
    "image": DatasetImage(),
    "problem": Value("string"),
    "solution": Value("string"),
})

"""
Note: To avoid excessive memory usage while debugging the pipeline,
we only build a small HF subset instead of all 88k examples.
You can increase `max_samples` later once everything runs end‑to‑end.
"""

MAX_SAMPLES = 5000  # temporary small subset for debugging

# Collect the SFT data into the hf_dict
with pushd(dataset_root):
    print("start to convert")
    for i, item in enumerate(tqdm(open_access_sft_data)):
        if i >= MAX_SAMPLES:
            break
        hf_dict["image"].append(load_image_from_path(item["image"]))
        hf_dict["problem"].append(item["problem"])
        hf_dict["solution"].append(item["solution"])

    # Place the SFT data into the 'train' split of a DatasetDict
    train_dataset = Dataset.from_dict(hf_dict, features=features)
    open_access_sft_dataset = DatasetDict({"train": train_dataset})

    print("start to save")
    # Save the dataset to disk
    open_access_sft_dataset.save_to_disk(save_path)


