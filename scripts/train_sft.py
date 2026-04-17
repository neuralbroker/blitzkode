#!/usr/bin/env python3
"""
Stage 2: Full SFT Training
Downloads base model and trains with LoRA on coding datasets.
"""

import os
import sys
import json
import torch
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "C:/Dev/Projects/BlitzKode/models"

BLITZKODE_BASE = Path("C:/Dev/Projects/BlitzKode")
HF_REPO = \"neuralbroker/blitzkode-1.5b-sft-v1\"
CHECKPOINT_DIR = BLITZKODE_BASE / "checkpoints" / "sft-1.5b-v1"
MODEL_NAME = "C:/Dev/Projects/BlitzKode/models/qwen1.5b"

def get_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory": props.total_memory / (1024**3),
            "free_memory": (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        }
    return None

def download_datasets():
    print("\n[LOADING DATASETS]")
    
    datasets = []
    
    # Local comprehensive coding problems
    problems = [
        # Arrays - Easy
        {"prompt": "Write a Python function to find the two sum of indices that add up to target.", 
         "response": "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n    return []\n```\n\nTime: O(n), Space: O(n)"},
        
        {"prompt": "Write a Python function to reverse an array in-place.",
         "response": "```python\ndef reverse(nums):\n    l, r = 0, len(nums) - 1\n    while l < r:\n        nums[l], nums[r] = nums[r], nums[l]\n        l += 1\n        r -= 1\n```\n\nTime: O(n), Space: O(1)"},
        
        {"prompt": "Write Python code to find maximum subarray sum (Kadane's algorithm).",
         "response": "```python\ndef max_subarray(nums):\n    max_sum = curr = nums[0]\n    for n in nums[1:]:\n        curr = max(n, curr + n)\n        max_sum = max(max_sum, curr)\n    return max_sum\n```\n\nTime: O(n), Space: O(1)"},
        
        # Strings
        {"prompt": "Write a Python function to check if a string is a palindrome.",
         "response": "```python\ndef is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n```\n\nTime: O(n), Space: O(n)"},
        
        {"prompt": "Write Python to implement string compression: 'aabbbc' -> 'a2b3c'",
         "response": "```python\ndef compress(s):\n    res = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            res.append(s[i-1] + str(count))\n            count = 1\n    res.append(s[-1] + str(count))\n    return ''.join(res)\n```"},
        
        # Linked Lists
        {"prompt": "Write Python to reverse a linked list.",
         "response": "```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    prev = None\n    while head:\n        next_node = head.next\n        head.next = prev\n        prev = head\n        head = next_node\n    return prev\n```\n\nTime: O(n), Space: O(1)"},
        
        # Trees
        {"prompt": "Write Python for binary tree inorder traversal (recursive).",
         "response": "```python\ndef inorder(root, res=[]):\n    if root:\n        inorder(root.left, res)\n        res.append(root.val)\n        inorder(root.right, res)\n    return res\n```\n\nTime: O(n), Space: O(n)"},
        
        {"prompt": "Write Python to find maximum depth of binary tree.",
         "response": "```python\ndef max_depth(root):\n    if not root:\n        return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))\n```\n\nTime: O(n), Space: O(n)"},
        
        # Dynamic Programming
        {"prompt": "Write Python for Fibonacci with memoization.",
         "response": "```python\ndef fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```\n\nTime: O(n), Space: O(n)"},
        
        {"prompt": "Write Python to solve coin change (minimum coins).",
         "response": "```python\ndef coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for c in coins:\n        for i in range(c, amount + 1):\n            dp[i] = min(dp[i], dp[i-c] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1\n```\n\nTime: O(n*amount), Space: O(amount)"},
        
        # Sorting
        {"prompt": "Write Python quicksort implementation.",
         "response": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```\n\nTime: O(n log n), Space: O(n)"},
        
        # Binary Search
        {"prompt": "Write Python binary search.",
         "response": "```python\ndef binary_search(arr, target):\n    l, r = 0, len(arr) - 1\n    while l <= r:\n        mid = (l + r) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            l = mid + 1\n        else:\n            r = mid - 1\n    return -1\n```\n\nTime: O(log n), Space: O(1)"},
        
        # Graphs
        {"prompt": "Write Python for BFS traversal.",
         "response": "```python\nfrom collections import deque\n\ndef bfs(graph, start):\n    visited = {start}\n    queue = deque([start])\n    result = []\n    while queue:\n        node = queue.popleft()\n        result.append(node)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return result\n```\n\nTime: O(V+E), Space: O(V)"},
        
        # Data Structures
        {"prompt": "Implement a Python stack using list.",
         "response": "```python\nclass Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        return self.items.pop() if self.items else None\n    \n    def peek(self):\n        return self.items[-1] if self.items else None\n    \n    def is_empty(self):\n        return len(self.items) == 0\n```"},
        
        {"prompt": "Implement a Python queue using collections.deque.",
         "response": "```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self):\n        self.items = deque()\n    \n    def enqueue(self, item):\n        self.items.append(item)\n    \n    def dequeue(self):\n        return self.items.popleft() if self.items else None\n    \n    def front(self):\n        return self.items[0] if self.items else None\n```"},
        
        # Hash Tables
        {"prompt": "Write Python to find longest substring without repeating characters.",
         "response": "```python\ndef length_of_longest_substring(s):\n    char_index = {}\n    max_len = start = 0\n    for i, c in enumerate(s):\n        if c in char_index and char_index[c] >= start:\n            start = char_index[c] + 1\n        char_index[c] = i\n        max_len = max(max_len, i - start + 1)\n    return max_len\n```\n\nTime: O(n), Space: O(min(n, alphabet))"},
        
        # Math
        {"prompt": "Write Python to check if a number is prime.",
         "response": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```\n\nTime: O(sqrt(n)), Space: O(1)"},
        
        # Two Pointers
        {"prompt": "Write Python for 3sum problem: find all triplets that sum to 0.",
         "response": "```python\ndef three_sum(nums):\n    res = []\n    nums.sort()\n    for i in range(len(nums) - 2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        l, r = i + 1, len(nums) - 1\n        while l < r:\n            s = nums[i] + nums[l] + nums[r]\n            if s == 0:\n                res.append([nums[i], nums[l], nums[r]])\n                l += 1\n                r -= 1\n                while l < r and nums[l] == nums[l-1]:\n                    l += 1\n            elif s < 0:\n                l += 1\n            else:\n                r -= 1\n    return res\n```\n\nTime: O(n^2), Space: O(n)"},
        
        # Sliding Window
        {"prompt": "Write Python for maximum sum subarray of size k.",
         "response": "```python\ndef max_sum_subarray(arr, k):\n    window_sum = sum(arr[:k])\n    max_sum = window_sum\n    for i in range(k, len(arr)):\n        window_sum += arr[i] - arr[i-k]\n        max_sum = max(max_sum, window_sum)\n    return max_sum\n```\n\nTime: O(n), Space: O(1)"},
        
        # Backtracking
        {"prompt": "Write Python to generate all permutations of a list.",
         "response": "```python\ndef permutations(nums):\n    res = []\n    def backtrack(path, used):\n        if len(path) == len(nums):\n            res.append(path[:])\n            return\n        for i, n in enumerate(nums):\n            if i in used:\n                continue\n            path.append(n)\n            used.add(i)\n            backtrack(path, used)\n            path.pop()\n            used.remove(i)\n    backtrack([], set())\n    return res\n```\n\nTime: O(n!*n), Space: O(n)"},
        
        # Greedy
        {"prompt": "Write Python for activity selection problem.",
         "response": "```python\ndef activity_selection(activities):\n    activities.sort(key=lambda x: x[1])\n    count = 1\n    end = activities[0][1]\n    for start, finish in activities[1:]:\n        if start >= end:\n            count += 1\n            end = finish\n    return count\n```\n\nTime: O(n log n), Space: O(1)"},
        
        # Recursion
        {"prompt": "Write Python to calculate factorial recursively.",
         "response": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```\n\nTime: O(n), Space: O(n)"},
        
        {"prompt": "Write Python to calculate power(x, n).",
         "response": "```python\ndef power(x, n):\n    if n == 0:\n        return 1\n    if n < 0:\n        return 1 / power(x, -n)\n    half = power(x, n // 2)\n    return half * half if n % 2 == 0 else half * half * x\n```\n\nTime: O(log n), Space: O(log n)"},
        
        # Design
        {"prompt": "Implement LRU Cache in Python.",
         "response": "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    \n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    \n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```\n\nTime: O(1), Space: O(capacity)"},
    ]
    
    datasets = problems
    print(f"  Using {len(datasets)} local coding problems")
    print(f"  Topics: Arrays, Strings, Linked Lists, Trees, DP, Graphs, etc.")
    
    return datasets

def main():
    print("=" * 60)
    print("STAGE 2: SFT TRAINING")
    print("=" * 60)
    
    gpu = get_gpu_info()
    if gpu:
        print(f"\n[GPU] {gpu['name']}")
        print(f"  VRAM: {gpu['total_memory']:.1f}GB total, {gpu['free_memory']:.1f}GB free")
    
    print("\n[CONFIG]")
    print("  Model: Qwen2.5-1.5B-Instruct (local)")
    print("  LoRA Rank: 32")
    print("  Batch: 2, Gradient Accum: 4")
    
    datasets = download_datasets()
    
    if not datasets:
        print("\n[ERROR] No datasets loaded")
        return
    
    print(f"\n[SAVING DATASET]")
    (BLITZKODE_BASE / "datasets" / "raw").mkdir(parents=True, exist_ok=True)
    dataset_file = BLITZKODE_BASE / "datasets" / "raw" / "blitzkode_sft_full.json"
    with open(dataset_file, "w") as f:
        json.dump(datasets, f, indent=2)
    print(f"  Saved to: {dataset_file}")
    
    print("\n[LOADING MODEL]")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    
    print(f"  Loading from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("  Model loaded!")
    
    print("\n[CONFIGURING LoRA]")
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
    print("  LoRA configured (r=32)")
    
    print("\n[CREATING TRAINER]")
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset
    
    def format_data(example):
        text = f"Problem: {example['prompt']}\n\nSolution: {example['response']}"
        return tokenizer(text, truncation=True, max_length=2048, padding=False)
    
    print("  Processing dataset...")
    hf_dataset = Dataset.from_list(datasets)
    hf_dataset = hf_dataset.map(format_data, remove_columns=["prompt", "response"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=hf_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    print("  Trainer ready!")
    
    print("\n[TRAINING...]")
    trainer.train()
    
    print("\n[SAVING]")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(CHECKPOINT_DIR / "final"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "final"))
    
    print(f"\n[SFT COMPLETE]")
    print(f"  Saved to: {CHECKPOINT_DIR / 'final'}")
    print("\nNEXT: python scripts/train_grpo.py")

if __name__ == "__main__":
    main()
