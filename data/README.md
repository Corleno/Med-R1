# Test Dataset

This folder contains small test datasets for development and collaboration purposes.

## Current Datasets

### OCT Train Test Dataset
- **Path**: `open_access_sft_data_hf_modality_OCT_train_test/`
- **Samples**: 100 (sampled from full dataset of 3,716 samples)
- **Size**: ~14MB
- **Purpose**: Testing and collaboration
- **Source**: Full dataset at `/data/datasets/OmniMedVQA/OmniMedVQA/open_access_sft_data_hf_modality_OCT_(Optical_Coherence_Tomography_train`

## Usage

To use this test dataset in training scripts:

```bash
--dataset_name /home/fayang/Med-R1/data/open_access_sft_data_hf_modality_OCT_train_test
```

## Creating Test Datasets

To create a new test dataset with a specific number of samples:

```bash
conda run -n med-r1 python scripts/create_test_dataset.py \
    --source /path/to/source/dataset \
    --output /home/fayang/Med-R1/data/your_test_dataset_name \
    --num_samples 100 \
    --seed 42
```

## Notes

- Test datasets are kept small (< 50MB) for git compatibility
- Full datasets remain in `/data/datasets/OmniMedVQA/OmniMedVQA/`
- Test datasets are sampled randomly with a fixed seed for reproducibility

