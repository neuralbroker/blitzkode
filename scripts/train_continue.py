#!/usr/bin/env python3
"""
Continue Training BlitzKode from DPO checkpoint
"""

import torch
from pathlib import Path
from datasets import load_dataset, Dataset

BLITZKODE_BASE = Path("C:/Dev/Projects/BlitzKode")
DPO_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "dpo-v1" / "final"
OUTPUT_DIR = BLITZKODE_BASE / "checkpoints" / "blitzkode-v2"

def format_sample(instruction, response):
    return {
        "text": f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    }

def load_datasets():
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    print("\n[1/1] Loading CodeAlpaca...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    formatted = [format_sample(s["instruction"], s["output"]) for s in ds]
    dataset = Dataset.from_list(formatted)
    print(f"  Loaded {len(formatted)} samples")
    print(f"\nTotal: {len(dataset)}")
    return dataset

def setup_model():
    print("\n[LOADING MODEL FROM DPO CHECKPOINT]")
    print(f"  Path: {DPO_CHECKPOINT}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, TaskType
    
    peft_config = PeftConfig.from_pretrained(str(DPO_CHECKPOINT))
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(DPO_CHECKPOINT),
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("  Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(DPO_CHECKPOINT))
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    print("="*60)
    print("BLITZKODE V2 - CONTINUE TRAINING")
    print("="*60)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    dataset = load_datasets()
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_32bit",
        max_steps=100,
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("\n[TRAINING...]")
    trainer.train()
    
    print("\n[SAVING MODEL]")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    print(f"\nModel saved to: {OUTPUT_DIR / 'final'}")
    print("\nDone! Run: python scripts/web_chat_v2.py")

if __name__ == "__main__":
    main()
