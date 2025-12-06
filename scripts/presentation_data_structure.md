# OmniMedVQA 数据处理结构

## 数据流程

### 基础训练流程 (SFT → GRPO)
```
原始 JSON 数据
    ↓ [data_process.py]
open_access_sft_data_hf/  (所有数据混合)
    ↓ [SFT 训练]
SFT 模型 checkpoint
    ↓ [GRPO 训练]
最终模型
```

### Cross-Modality/Cross-Task 实验流程
```
原始 JSON 数据
    ↓ [data_process_split.py]
按模态/任务分组 + 80/20 split
    ↓
26 个独立数据集
    ↓ [实验训练]
Cross-modality/cross-task 评估
```

---

## 数据集存储结构

### 1. 基础数据集

用于 SFT + GRPO 基础训练流程。

路径: `/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf/`

内容:
- 所有模态和任务类型的数据混合
- 图片已处理 (384×384)
- 文本已格式化
- HuggingFace Dataset 格式

文件夹结构:
```
open_access_sft_data_hf/
├── dataset_dict.json          # 数据集元信息
└── train/
    ├── data-*.arrow          # 处理好的数据（图片+文本，Arrow格式）
    ├── dataset_info.json     # 数据集信息（features, 样本数等）
    └── state.json            # 状态信息
```

在训练脚本中的使用:
- SFT: `configs/qwen2_5/qwen2_5vl_sft_config_*.yaml` (第10行)
- GRPO: `scripts/grpo_script.sh` (第45行)

---

### 2. Cross-Modality 数据集 (8个模态，16个数据集)

用于 cross-modality 实验：训练一个模态，测试其他7个模态。

| 模态 | 训练集 | 测试集 |
|------|--------|--------|
| CT | `open_access_sft_data_hf_modality_CT_train/` | `open_access_sft_data_hf_modality_CT_test/` |
| MRI | `open_access_sft_data_hf_modality_MRI_train/` | `open_access_sft_data_hf_modality_MRI_test/` |
| X-Ray | `open_access_sft_data_hf_modality_X_Ray_train/` | `open_access_sft_data_hf_modality_X_Ray_test/` |
| Ultrasound | `open_access_sft_data_hf_modality_Ultrasound_train/` | `open_access_sft_data_hf_modality_Ultrasound_test/` |
| Dermoscopy | `open_access_sft_data_hf_modality_Dermoscopy_train/` | `open_access_sft_data_hf_modality_Dermoscopy_test/` |
| Fundus | `open_access_sft_data_hf_modality_Fundus_train/` | `open_access_sft_data_hf_modality_Fundus_test/` |
| OCT | `open_access_sft_data_hf_modality_OCT_train/` | `open_access_sft_data_hf_modality_OCT_test/` |
| Microscopy | `open_access_sft_data_hf_modality_Microscopy_train/` | `open_access_sft_data_hf_modality_Microscopy_test/` |

Split 比例: 80% train, 20% test (每个模态独立 split)

**注意**: 
- 训练时使用 `train` split（HF格式）
- 评估时使用 JSON 格式的测试数据（见"测试数据集 (JSON 格式)"章节），而不是 HF 格式的 `test` split
- HF 格式的 `test` split **目前未被使用**，保留作为数据备份

---

### 3. Cross-Task 数据集 (5个任务，10个数据集)

用于 cross-task 实验：训练一个任务，测试其他4个任务。

| 任务类型 | 训练集 | 测试集 |
|---------|--------|--------|
| Anatomy Identification | `open_access_sft_data_hf_task_Anatomy_Identification_train/` | `open_access_sft_data_hf_task_Anatomy_Identification_test/` |
| Disease Diagnosis | `open_access_sft_data_hf_task_Disease_Diagnosis_train/` | `open_access_sft_data_hf_task_Disease_Diagnosis_test/` |
| Lesion Grading | `open_access_sft_data_hf_task_Lesion_Grading_train/` | `open_access_sft_data_hf_task_Lesion_Grading_test/` |
| Modality Recognition | `open_access_sft_data_hf_task_Modality_Recognition_train/` | `open_access_sft_data_hf_task_Modality_Recognition_test/` |
| Other Biological Attributes | `open_access_sft_data_hf_task_Other_Biological_Attributes_train/` | `open_access_sft_data_hf_task_Other_Biological_Attributes_test/` |

Split 比例: 80% train, 20% test (每个任务独立 split)

**注意**: 
- 训练时使用 `train` split（HF格式）
- 评估时使用 JSON 格式的测试数据（见"测试数据集 (JSON 格式)"章节），而不是 HF 格式的 `test` split
- HF 格式的 `test` split **目前未被使用**，保留作为数据备份

---

### 4. 测试数据集 (JSON 格式，用于评估)

用于模型评估脚本，从 HF 格式的 test split 转换而来。

路径: `/data/datasets/OmniMedVQA/OmniMedVQA/eval_json/`

文件夹结构:
```
eval_json/
├── modality/
│   ├── CT(Computed_Tomography)_test.json
│   ├── MRI_test.json
│   ├── X_Ray_test.json
│   ├── Ultrasound_test.json
│   ├── Dermoscopy_test.json
│   ├── Fundus_test.json
│   ├── OCT_test.json
│   └── Microscopy_test.json
└── task/
    ├── Anatomy_Identification_test.json
    ├── Disease_Diagnosis_test.json
    ├── Lesion_Grading_test.json
    ├── Modality_Recognition_test.json
    └── Other_Biological_Attributes_test.json
```

数据格式:
- JSON 数组，每个元素包含:
  - `image`: 图片路径 (相对路径，如 `Images/Chest CT Scan/test/...`)
  - `problem`: 问题文本 (包含选项)
  - `solution`: 答案 (格式: `<answer> X </answer>`)

用途:
- 评估脚本 (`src/eval_vqa/test_qwen2_5vl_vqa_nothink.py`) 使用这些 JSON 文件进行模型评估
- 与 HF 格式的 test split 数据一致，只是格式不同（JSON vs HuggingFace Dataset）
- **用途**: 训练后的模型性能评估（evaluation）
- **与 HF test folder 的区别**: HF test folder 目前未被使用，JSON 格式用于训练后的评估

---

## 数据读取位置

### SFT 训练

脚本: `scripts/sft_vqa_2_5.sh`

配置文件: `src/r1-v/configs/qwen2_5/qwen2_5vl_sft_config_*.yaml`
```yaml
dataset_name: /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf
```

代码位置: `src/r1-v/src/open_r1/sft_2_5.py:212`
```python
dataset = load_from_disk(script_args.dataset_name)
```

### GRPO 训练

脚本: `scripts/grpo_script.sh`

命令行参数 (第45行):
```bash
--dataset_name /data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf
```

代码位置: `src/r1-v/src/open_r1/grpo_vqa_nothink.py`
```python
# Loads via load_from_disk()
```

### 模型评估

脚本: `scripts/eval_qwen2_5vl_nothink_CT.sh`

命令行参数:
```bash
--prompt_path /data/datasets/OmniMedVQA/OmniMedVQA/eval_json/modality/CT(Computed_Tomography)_test.json
--image_folder /data/datasets/OmniMedVQA/OmniMedVQA
```

代码位置: `src/eval_vqa/test_qwen2_5vl_vqa_nothink.py:44-45`
```python
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  # 加载 JSON 格式的测试数据
```

---

## 数据统计

### 模态分布
- CT: 15,808 samples
- MRI: 31,877 samples  
- X-Ray: 7,916 samples
- Ultrasound: 10,991 samples
- Dermoscopy: 6,679 samples
- Fundus: 5,398 samples
- OCT: 4,646 samples
- Microscopy: 5,680 samples
- 总计: 88,996 samples

### 任务类型分布
- Anatomy Identification: 16,448 samples
- Disease Diagnosis: 55,387 samples
- Lesion Grading: 2,098 samples
- Modality Recognition: 11,565 samples
- Other Biological Attributes: 3,498 samples
- 总计: 88,996 samples

---

## 数据格式

每个数据集包含：
- `image`: PIL Image (384×384, RGB)
- `problem`: string (问题 + 选项)
- `solution`: string (答案，格式: `<answer> X </answer>`)

所有数据集均为 HuggingFace Dataset 格式，可直接用于训练。

### 存储格式说明

每个数据集文件夹包含：
- `dataset_dict.json`: 定义数据集的 split（train/test）
- `train/` 或 `test/` 文件夹: 包含实际数据
  - `data-*.arrow`: 实际数据文件（Apache Arrow 格式，包含图片和文本）
  - `dataset_info.json`: 数据字段定义（image, problem, solution）
  - `state.json`: 数据文件列表和版本信息

Arrow 格式的优势：高效存储、快速加载、支持按需读取，适合大规模数据集。

---

## 使用场景

1. **基础训练流程** (SFT → GRPO):
   - 使用: `open_access_sft_data_hf/`
   - 所有数据混合，用于通用训练

2. **Cross-modality 实验**:
   - 训练: 使用单个模态的 `train` split (例如: `open_access_sft_data_hf_modality_CT_train/`)
   - 测试: 使用其他7个模态的 `test` split

3. **Cross-task 实验**:
   - 训练: 使用单个任务的 `train` split (例如: `open_access_sft_data_hf_task_Disease_Diagnosis_train/`)
   - 测试: 使用其他4个任务的 `test` split

4. **模型评估**:
   - 使用: `eval_json/{modality|task}/*_test.json`
   - JSON 格式的测试数据，包含图片路径、问题、答案
   - 用于训练后的模型性能评估

---

## 总结

- 训练数据集数: 14个 (1个基础 + 8个模态train + 5个任务train)
- 测试数据集数: 13个 (8个模态test + 5个任务test)
  - HF格式: 目前未被使用，保留作为数据备份
  - JSON格式: 用于训练后的模型评估（evaluation）
- 训练数据格式: HuggingFace Dataset (Arrow 格式)
- 测试数据格式: 
  - HuggingFace Dataset (Arrow 格式，目前未被使用)
  - JSON 格式 (用于训练后评估)
- 数据状态: 已处理完成，可直接用于训练和评估
- 存储位置: `/data/datasets/OmniMedVQA/OmniMedVQA/`

