"""
Retrieve all json results from the results directory and print the accuracy for each file

Example usage: 
SFT: python scripts/others/retrieve_results.py /mnt/task_runtime/results/Qwen3-VL-2B-Instruct-SFT-MRI-600
GRPO (think): python scripts/others/retrieve_results.py /mnt/task_runtime/results/Qwen3-VL-2B-GRPO-MRI-600-think
GRPO (nothink): python scripts/others/retrieve_results.py /mnt/task_runtime/results/Qwen3-VL-2B-GRPO-MRI-600-nothink
"""

import os
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python retrieve_results.py <results_dir>")
    sys.exit(1)
results_dir = sys.argv[1]

import csv

csv_output_path = os.path.join(results_dir, "accuracy_results.csv")
with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file", "accuracy"])
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), "r") as f:
                data = json.load(f)
                # get accuracy from the data
                accuracy = round(data["accuracy"], 2)
                writer.writerow([file, accuracy])

print(f"Accuracy results saved to {csv_output_path}")