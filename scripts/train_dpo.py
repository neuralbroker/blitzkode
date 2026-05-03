#!/usr/bin/env python3
"""
Stage 3: DPO Training (Direct Preference Optimization)
Uses TRL's DPOTrainer for preference-based training.
"""

import os
import sys
import json
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BLITZKODE_BASE = SCRIPT_DIR.parent

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(BLITZKODE_BASE / "models")

GRPO_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "grpo-v1" / "final"
DPO_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "dpo-v1"

def get_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory": props.total_memory / (1024**3),
            "free_memory": (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        }
    return None

def get_dpo_dataset():
    """
    Generate DPO preference pairs.
    Each entry has a 'chosen' (preferred) and 'rejected' (less preferred) response.
    """
    preference_data = [
        {
            "prompt": "Write a Python function to find the two sum of indices that add up to target.",
            "chosen": """```python
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
```

**Explanation:**
- Use a hash map to store each number and its index
- For each number, check if (target - num) exists in the map
- Time: O(n), Space: O(n)""",
            "rejected": """```python
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```"""
        },
        {
            "prompt": "Write Python code to find maximum subarray sum (Kadane's algorithm).",
            "chosen": """```python
def max_subarray(nums):
    max_sum = curr = nums[0]
    for n in nums[1:]:
        curr = max(n, curr + n)
        max_sum = max(max_sum, curr)
    return max_sum
```

**Kadane's Algorithm:**
- Keep track of current subarray sum and max so far
- Reset if current becomes negative
- Time: O(n), Space: O(1)""",
            "rejected": """```python
def max_subarray(nums):
    max_sum = nums[0]
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            sub_sum = sum(nums[i:j+1])
            max_sum = max(max_sum, sub_sum)
    return max_sum
```"""
        },
        {
            "prompt": "Write a Python function to check if a string is a palindrome.",
            "chosen": """```python
def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
```

**Approach:**
- Remove all non-alphanumeric characters
- Convert to lowercase
- Compare with reversed string
- Time: O(n), Space: O(n)""",
            "rejected": """```python
def is_palindrome(s):
    return s == s[::-1]
```"""
        },
        {
            "prompt": "Write Python quicksort implementation.",
            "chosen": """```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

**How it works:**
- Choose middle element as pivot
- Partition into < pivot, = pivot, > pivot
- Recursively sort left and right
- Time: O(n log n) avg, O(n²) worst, Space: O(n)""",
            "rejected": """```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    return quicksort([x for x in arr if x < arr[0]]) + [x for x in arr if x == arr[0]] + quicksort([x for x in arr if x > arr[0]])
```"""
        },
        {
            "prompt": "Implement LRU Cache in Python.",
            "chosen": """```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

**Key Points:**
- OrderedDict maintains insertion order
- move_to_end updates access order
- popitem removes oldest when full
- All operations: O(1)""",
            "rejected": """```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key, -1)
    
    def put(self, key, value):
        if key in self.cache:
            del self.cache[key]
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            del self.cache[next(iter(self.cache))]
```"""
        },
        {
            "prompt": "Write Python binary search.",
            "chosen": """```python
def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1
```

**Algorithm:**
- Maintain left and right bounds
- Check middle element each iteration
- Narrow search space in half
- Time: O(log n), Space: O(1)""",
            "rejected": """```python
def binary_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```"""
        },
        {
            "prompt": "Write Python for BFS traversal.",
            "chosen": """```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result
```

**How it works:**
- Use queue for FIFO processing
- Track visited nodes to avoid cycles
- Process level by level
- Time: O(V + E), Space: O(V)""",
            "rejected": """```python
def bfs(graph, start):
    result = [start]
    for neighbor in graph[start]:
        result.append(neighbor)
    return result
```"""
        },
        {
            "prompt": "Write Python for Fibonacci with memoization.",
            "chosen": """```python
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

**Memoization:**
- Cache computed results
- Avoid redundant calculations
- Top-down dynamic programming
- Time: O(n), Space: O(n)""",
            "rejected": """```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```"""
        },
        {
            "prompt": "Write Python to reverse a linked list.",
            "chosen": """```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev
```

**Iterative approach:**
- Reverse pointers one by one
- Keep track of previous node
- Time: O(n), Space: O(1)""",
            "rejected": """```python
def reverse_list(head):
    if not head:
        return None
    new_head = reverse_list(head.next)
    head.next.next = head
    head.next = None
    return new_head
```"""
        },
        {
            "prompt": "Write Python to find maximum depth of binary tree.",
            "chosen": """```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**Recursive solution:**
- Base case: empty tree = depth 0
- Recursive: 1 + max(depth of left, depth of right)
- Time: O(n), Space: O(n)""",
            "rejected": """```python
def max_depth(root):
    count = 0
    while root:
        count += 1
        root = root.left
    return count
```"""
        },
    ]
    
    return preference_data

def main():
    global GRPO_CHECKPOINT
    
    print("=" * 60)
    print("STAGE 3: DPO TRAINING")
    print("=" * 60)
    
    gpu = get_gpu_info()
    if gpu:
        print(f"\n[GPU] {gpu['name']}")
        print(f"  VRAM: {gpu['total_memory']:.1f}GB total, {gpu['free_memory']:.1f}GB free")
    
    print("\n[CONFIG]")
    print("  Base Model: Qwen2.5-1.5B-Instruct")
    print("  Previous Stage: GRPO")
    print("  Beta: 0.1 (DPO loss coefficient)")
    
    if not GRPO_CHECKPOINT.exists():
        print(f"\n[WARNING] GRPO checkpoint not found: {GRPO_CHECKPOINT}")
        print("Falling back to SFT checkpoint...")
        GRPO_CHECKPOINT = BLITZKODE_BASE / "checkpoints" / "sft-1.5b-v1" / "final"
        if not GRPO_CHECKPOINT.exists():
            print(f"[ERROR] SFT checkpoint not found either")
            print("Run: python scripts/train_sft.py")
            return
    
    print(f"\n[LOADING MODEL FROM {GRPO_CHECKPOINT.name}]")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType, PeftConfig
    from trl import DPOTrainer, DPOConfig
    
    peft_config = PeftConfig.from_pretrained(str(GRPO_CHECKPOINT))
    peft_config.inference_mode = False
    
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        target_modules=peft_config.target_modules,
        lora_dropout=peft_config.lora_dropout,
        bias=peft_config.bias,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    
    from safetensors.torch import load_file
    adapter_state = load_file(str(GRPO_CHECKPOINT / "adapter_model.safetensors"))
    model.load_state_dict(adapter_state, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(str(GRPO_CHECKPOINT), trust_remote_code=True)
    print("  Model loaded!")
    
    print("\n[LOADING PREFERENCE DATASET]")
    dpo_data = get_dpo_dataset()
    from datasets import Dataset
    hf_dataset = Dataset.from_list(dpo_data)
    print(f"  Loaded {len(dpo_data)} preference pairs")
    
    print("\n[CREATING DPO TRAINER]")
    dpo_config = DPOConfig(
        output_dir=str(DPO_CHECKPOINT),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        beta=0.1,
        gradient_checkpointing=False,
    )
    
    model.warnings_issued = {}
    model.train()
    model.base_model.train()
    
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
    
    print("  DPO Trainer ready!")
    print("\n  Preference pairs:")
    print("    - Chosen: Better solution with explanation")
    print("    - Rejected: Simpler/less optimal solution")
    
    print("\n[DPO TRAINING...]")
    trainer.train()
    
    print("\n[SAVING]")
    DPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(DPO_CHECKPOINT / "final"))
    tokenizer.save_pretrained(str(DPO_CHECKPOINT / "final"))
    
    print(f"\n[COMPLETE]")
    print(f"  Saved to: {DPO_CHECKPOINT / 'final'}")
    print("\nNEXT: python scripts/export_gguf.py")

if __name__ == "__main__":
    main()
