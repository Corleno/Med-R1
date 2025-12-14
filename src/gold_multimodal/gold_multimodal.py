# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# Full training:
python -m src.gold_multimodal.gold_multimodal \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --dataset_name data/open_access_sft_data_hf_modality_OCT_train_test \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing

# LoRA:

# 8 GPUs LoRA training
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  -m src.gold_multimodal.gold_multimodal \
  --deepspeed src/training_configs/zero3.json \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --dataset_name data/open_access_sft_data_hf_modality_OCT_train_test \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --output_dir gold-model \
  --num_train_epochs 1 \
  --push_to_hub \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 16 \
  --gradient_checkpointing false\
  --report_to wandb \
  --run_name Qwen2.5-1.5B-Instruct-to-Qwen3-4B-Instruct-2507-Gold-8GPUs-LoRA-64
"""

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GenerationConfig, AutoProcessor

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from .gold_multimodal_config import GOLDMultimodalConfig
from .gold_multimodal_trainer import GOLDMultimodalTrainer


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDMultimodalConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=training_args.student_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_tokenizer_name_or_path is None and training_args.use_uld_loss:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path
    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        # use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     revision=model_args.model_revision,
    #     trust_remote_code=model_args.trust_remote_code,
    #     padding_side="left",
    # )
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    ################
    # Dataset
    ################
    if training_args.dataset_from_disk:
        dataset_dict = load_from_disk(script_args.dataset_name)
    else:
        dataset_dict = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    if training_args.dataset_type == "vqa":
        QUESTION_TEMPLATE = "{Question} Your task: 1. Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. 2. No extra information or text outside of this tag."

        def make_conversation_image(example):
            example["messages"] = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text", 
                                "text": QUESTION_TEMPLATE.format(Question=example["problem"]),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text", 
                                "text": example["solution"],
                            },
                        ],
                    },
                ]
            return example

        train_dataset = dataset_dict[script_args.dataset_train_split]
        if "image" in train_dataset.features:
            train_dataset = train_dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        else:
            raise ValueError("no image in dataset")

        if training_args.eval_strategy != "no":
            eval_dataset = dataset_dict[script_args.dataset_test_split]
            if "image" in eval_dataset.features:
                eval_dataset = eval_dataset.map(make_conversation_image)
            else:
                raise ValueError("no image in dataset")
        else:
            eval_dataset = None

    ################
    # Training
    ################
    trainer = GOLDMultimodalTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)