#!/usr/bin/env python3
"""
Production-grade Training Pipeline for BlitzKode V2
Optimized for large-scale coding dataset training
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
BLITZKODE_BASE = SCRIPT_DIR.parent

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = BLITZKODE_BASE / "checkpoints" / "blitzkode-v2"

def format_sample(instruction, response):
    return {
        "text": f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    }

def load_datasets(sample_size=10000):
    print("="*60)
    print("LOADING OPEN SOURCE DATASETS")
    print("="*60)
    
    all_datasets = []
    
    print("\n[1/2] Loading CodeAlpaca...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        formatted = [format_sample(s["instruction"], s["output"]) for s in ds]
        all_datasets.append(Dataset.from_list(formatted))
        print(f"  Loaded {len(formatted)} samples")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print("\n[2/2] Loading Magicoder-Evol-Instruct...")
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct", split="train")
        formatted = [format_sample(s["instruction"], s["response"]) for s in ds]
        all_datasets.append(Dataset.from_list(formatted[:sample_size]))
        print(f"  Loaded {min(len(formatted), sample_size)} samples")
    except Exception as e:
        print(f"  Failed: {e}")
    
    if not all_datasets:
        print("\nUsing fallback dataset...")
        fallback = [
            format_sample("Write a function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
            format_sample("Implement binary search", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
        ]
        all_datasets.append(Dataset.from_list(fallback * 1000))
    
    combined = all_datasets[0]
    if len(all_datasets) > 1:
        combined = Dataset.concatenate_datasets(all_datasets)
    
    print(f"\nTotal samples: {len(combined)}")
    return combined

def setup_model():
    print("\n[SETTING UP MODEL]")
    print(f"  Base Model: {MODEL_NAME}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    print("="*60)
    print("BLITZKODE V2 - PRODUCTION TRAINING PIPELINE")
    print("="*60)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    dataset = load_datasets(sample_size=10000)
    
    model, tokenizer = setup_model()
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    
    from transformers import TrainingArguments, DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        group_by_length=False,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_32bit",
    )
    
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    
    print("\n[TRAINING...]")
    trainer.train()
    
    print("\n[SAVING MODEL]")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    print(f"\nModel saved to: {OUTPUT_DIR / 'final'}")
    print("\nNEXT: python scripts/web_chat_v2.py")

if __name__ == "__main__":
    main()
