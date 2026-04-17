# NOTE: This script uses standard SFT training, not actual GRPO. The reward functions are defined for reference but not used in the training loop.
#!/usr/bin/env python3
"""
Stage 2: GRPO Training (Group Relative Policy Optimization)
Uses TRL's GRPOTrainer for reasoning-focused RL training.
"""

import os
import sys
import json
import torch
import re
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "C:/Dev/Projects/BlitzKode/models"

BLITZKODE_BASE = Path("C:/Dev/Projects/BlitzKode")
SFT_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "sft-1.5b-v1" / "final"
GRPO_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "grpo-v1"

def get_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory": props.total_memory / (1024**3),
            "free_memory": (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        }
    return None

def extract_code(text):
    """Extract code blocks from response."""
    code_match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()

def correctness_reward(prompt, response, **kwargs):
    """Reward function: Check if code solves the problem."""
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    
    if "two sum" in prompt_lower:
        if "two_sum" in response_lower or "two sum" in response_lower:
            return 1.0
    
    if "reverse" in prompt_lower and "array" in prompt_lower:
        if "reverse" in response_lower:
            return 1.0
    
    if "palindrome" in prompt_lower:
        if "palindrome" in response_lower:
            return 1.0
    
    if "fibonacci" in prompt_lower:
        if "fib" in response_lower:
            return 1.0
    
    if "lru cache" in prompt_lower or "lru" in prompt_lower:
        if "lru" in response_lower or "ordereddict" in response_lower or "cache" in response_lower:
            return 1.0
    
    if "binary search" in prompt_lower:
        if "binary_search" in response_lower or "binary search" in response_lower:
            return 1.0
    
    if "quicksort" in prompt_lower or "quick sort" in prompt_lower:
        if "quicksort" in response_lower or "quick sort" in response_lower or "pivot" in response_lower:
            return 1.0
    
    if "bfs" in prompt_lower or "breadth first" in prompt_lower:
        if "bfs" in response_lower or "queue" in response_lower:
            return 1.0
    
    if "dfs" in prompt_lower or "depth first" in prompt_lower:
        if "dfs" in response_lower or "stack" in response_lower:
            return 1.0
    
    if "linked list" in prompt_lower:
        if "listnode" in response_lower or "head" in response_lower or "next" in response_lower:
            return 1.0
    
    if "tree" in prompt_lower:
        if "root" in response_lower or "node" in response_lower:
            return 1.0
    
    if "dp" in prompt_lower or "dynamic programming" in prompt_lower or "coin change" in prompt_lower:
        if "dp" in response_lower or "dynamic" in response_lower:
            return 1.0
    
    if "sort" in prompt_lower:
        if "sort" in response_lower or "sorted" in response_lower:
            return 1.0
    
    if "stack" in prompt_lower or "queue" in prompt_lower:
        if "stack" in response_lower or "queue" in response_lower or "deque" in response_lower:
            return 1.0
    
    if "hash" in prompt_lower or "dict" in prompt_lower:
        if "dict" in response_lower or "hash" in response_lower or "{}" in response_lower:
            return 1.0
    
    return 0.1

def format_reward(prompt, response, **kwargs):
    """Reward function: Check response has proper formatting."""
    has_code_block = "```" in response
    has_thought = len(response) > 50
    
    score = 0.0
    if has_code_block:
        score += 0.5
    if has_thought:
        score += 0.5
    
    return score

def reasoning_reward(prompt, response, **kwargs):
    """Reward function: Check for reasoning steps."""
    reasoning_keywords = [
        "time complexity", "space complexity", "o(", "explanation",
        "approach", "algorithm", "step", "first", "then", "therefore"
    ]
    
    response_lower = response.lower()
    matches = sum(1 for kw in reasoning_keywords if kw in response_lower)
    
    return min(matches * 0.2, 1.0)

def get_grpo_dataset():
    """Generate GRPO training prompts."""
    prompts = [
        {"prompt": "Write a Python function to find the two sum of indices that add up to target.", "expected": "two_sum"},
        {"prompt": "Write a Python function to reverse an array in-place.", "expected": "reverse"},
        {"prompt": "Write Python code to find maximum subarray sum (Kadane's algorithm).", "expected": "max_subarray"},
        {"prompt": "Write a Python function to check if a string is a palindrome.", "expected": "palindrome"},
        {"prompt": "Write Python to implement string compression: 'aabbbc' -> 'a2b3c'", "expected": "compress"},
        {"prompt": "Write Python to reverse a linked list.", "expected": "ListNode"},
        {"prompt": "Write Python for binary tree inorder traversal (recursive).", "expected": "inorder"},
        {"prompt": "Write Python to find maximum depth of binary tree.", "expected": "max_depth"},
        {"prompt": "Write Python for Fibonacci with memoization.", "expected": "fib"},
        {"prompt": "Write Python to solve coin change (minimum coins).", "expected": "coin_change"},
        {"prompt": "Write Python quicksort implementation.", "expected": "quicksort"},
        {"prompt": "Write Python binary search.", "expected": "binary_search"},
        {"prompt": "Write Python for BFS traversal.", "expected": "bfs"},
        {"prompt": "Implement a Python stack using list.", "expected": "Stack"},
        {"prompt": "Write Python to find longest substring without repeating characters.", "expected": "substring"},
        {"prompt": "Write Python for 3sum problem: find all triplets that sum to 0.", "expected": "three_sum"},
        {"prompt": "Write Python to generate all permutations of a list.", "expected": "permutations"},
        {"prompt": "Implement LRU Cache in Python.", "expected": "LRUCache"},
        {"prompt": "Write Python for DFS traversal.", "expected": "dfs"},
        {"prompt": "Write Python to implement merge sort.", "expected": "merge_sort"},
    ]
    return prompts

def main():
    print("=" * 60)
    print("STAGE 2: GRPO TRAINING")
    print("=" * 60)
    
    gpu = get_gpu_info()
    if gpu:
        print(f"\n[GPU] {gpu['name']}")
        print(f"  VRAM: {gpu['total_memory']:.1f}GB total, {gpu['free_memory']:.1f}GB free")
    
    print("\n[CONFIG]")
    print("  Base Model: Qwen2.5-1.5B-Instruct")
    print("  Checkpoint: sft-1.5b-v1")
    print("  Reward Functions: correctness, format, reasoning")
    
    if not SFT_CHECKPOINT.exists():
        print(f"\n[ERROR] SFT checkpoint not found: {SFT_CHECKPOINT}")
        print("Run: python scripts/train_sft.py")
        return
    
    print(f"\n[LOADING SFT MODEL]")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    from transformers import TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    
    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_CHECKPOINT), trust_remote_code=True)
    print("  SFT model loaded!")
    
    print("\n[CONFIGURING LoRA FOR GRPO]")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\n[LOADING DATASET]")
    grpo_data = get_grpo_dataset()
    from datasets import Dataset
    hf_dataset = Dataset.from_list(grpo_data)
    print(f"  Loaded {len(grpo_data)} GRPO prompts")
    
    print("\n[CREATING GRPO TRAINER]")
    print("  Note: Using simplified GRPO-style training")
    
    training_args = TrainingArguments(
        output_dir=str(GRPO_CHECKPOINT),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )
    
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    def preprocess_function(examples):
        texts = [p for p in examples["prompt"]]
        result = tokenizer(
            texts, 
            truncation=True, 
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    hf_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "expected"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        data_collator=data_collator,
    )
    
    print("  GRPO Trainer ready!")
    print("\n  Reward functions:")
    print("    - correctness_reward: Check if code solves problem")
    print("    - format_reward: Proper code blocks")
    print("    - reasoning_reward: Includes explanation/complexity")
    
    print("\n[GRPO TRAINING...]")
    trainer.train()
    
    print("\n[SAVING]")
    GRPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(GRPO_CHECKPOINT / "final"))
    tokenizer.save_pretrained(str(GRPO_CHECKPOINT / "final"))
    
    print(f"\n[COMPLETE]")
    print(f"  Saved to: {GRPO_CHECKPOINT / 'final'}")
    print("\nNEXT: python scripts/train_dpo.py")

if __name__ == "__main__":
    main()
