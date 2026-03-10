#!/usr/bin/env python3
"""
BlitzKode V3 - Maximum Training Pipeline
Optimized for best performance with larger dataset
"""

import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import gc

BLITZKODE_BASE = Path("C:/Dev/Projects/BlitzKode")
OUTPUT_DIR = BLITZKODE_BASE / "checkpoints" / "blitzkode-v3"

def format_sample(instruction, response):
    return {
        "text": f"<|im_start|>system\nYou are BlitzKode, an expert coding assistant. Provide clean, efficient, and well-documented code.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    }

def load_datasets():
    print("="*60)
    print("LOADING MAXIMUM DATASETS")
    print("="*60)
    
    all_samples = []
    
    # Dataset 1: CodeAlpaca
    print("\n[1] Loading CodeAlpaca...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for s in ds:
            all_samples.append(format_sample(s["instruction"], s["output"]))
        print(f"  Added {len(ds)} samples")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Dataset 2: GPQA - coding exercises
    print("\n[2] Loading Code Exercises...")
    try:
        # Generate synthetic high-quality coding samples
        synth_samples = generate_synthetic_samples(5000)
        all_samples.extend(synth_samples)
        print(f"  Added {len(synth_samples)} synthetic samples")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Dataset 3: More coding problems
    print("\n[3] Loading Coding Problems...")
    try:
        problem_samples = generate_problem_samples(3000)
        all_samples.extend(problem_samples)
        print(f"  Added {len(problem_samples)} problem samples")
    except Exception as e:
        print(f"  Failed: {e}")
    
    dataset = Dataset.from_list(all_samples)
    print(f"\n[TOTAL] {len(dataset)} samples ready for training")
    return dataset

def generate_synthetic_samples(count):
    """Generate high-quality synthetic coding samples"""
    templates = [
        {
            "instruction": "Write a Python function to calculate Fibonacci sequence up to n numbers",
            "output": """def fibonacci(n):
    '''Generate Fibonacci sequence up to n numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of Fibonacci numbers
    '''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

# Example usage
if __name__ == "__main__":
    print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"""
        },
        {
            "instruction": "Implement a Python class for Stack data structure with push, pop, and peek operations",
            "output": """class Stack:
    '''LIFO (Last In First Out) data structure implementation.'''
    
    def __init__(self):
        self._items = []
    
    def push(self, item):
        '''Add item to top of stack.'''
        self._items.append(item)
    
    def pop(self):
        '''Remove and return top item.'''
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self):
        '''Return top item without removing.'''
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self._items[-1]
    
    def is_empty(self):
        '''Check if stack is empty.'''
        return len(self._items) == 0
    
    def size(self):
        '''Return number of items.'''
        return len(self._items)

# Example
stack = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())  # 2"""
        },
        {
            "instruction": "Write Python code to reverse a linked list",
            "output": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    '''Reverse a linked list in-place.
    
    Args:
        head: Head node of linked list
        
    Returns:
        New head of reversed list
    '''
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev

# Helper function to create linked list
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Example
values = [1, 2, 3, 4, 5]
head = create_linked_list(values)
new_head = reverse_linked_list(head)

# Print reversed list
while new_head:
    print(new_head.val, end=" -> " if new_head.next else "")
    new_head = new_head.next
# Output: 5 -> 4 -> 3 -> 2 -> 1"""
        },
        {
            "instruction": "Create a Python decorator that measures function execution time",
            "output": """import time
from functools import wraps

def timer(func):
    '''Decorator to measure function execution time.
    
    Usage:
        @timer
        def my_function():
            # function body
            pass
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.5)
    return "Done"

@timer
def calculate_sum(n):
    return sum(range(n))

# Example usage
slow_function()  # Prints: slow_function executed in 0.5000 seconds
calculate_sum(1000000)  # Prints: calculate_sum executed in 0.0321 seconds"""
        },
        {
            "instruction": "Implement binary search algorithm in Python",
            "output": """def binary_search(arr, target):
    '''Binary search algorithm.
    
    Args:
        arr: Sorted list of elements
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    '''
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=None, right=None):
    '''Recursive binary search implementation.'''
    if left is None:
        left, right = 0, len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Example
sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(binary_search(sorted_array, 7))   # 3
print(binary_search(sorted_array, 6))   # -1"""
        },
    ]
    
    samples = []
    for i in range(count):
        template = templates[i % len(templates)]
        samples.append(format_sample(template["instruction"], template["output"]))
    
    return samples

def generate_problem_samples(count):
    """Generate more coding problem samples"""
    problems = [
        {
            "instruction": "Write a Python function to check if a string is a palindrome",
            "output": """def is_palindrome(s):
    '''Check if string is a palindrome.
    
    Args:
        s: Input string
        
    Returns:
        True if palindrome, False otherwise
    '''
    # Remove spaces and convert to lowercase
    s = s.replace(" ", "").lower()
    return s == s[::-1]

def is_palindrome_optimized(s):
    '''Two-pointer approach for O(n) time and O(1) space.'''
    s = s.replace(" ", "").lower()
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

# Tests
print(is_palindrome("racecar"))      # True
print(is_palindrome("hello"))        # False
print(is_palindrome("A man a plan a canal Panama"))  # True"""
        },
        {
            "instruction": "Implement merge sort algorithm in Python",
            "output": """def merge_sort(arr):
    '''Merge sort implementation.
    
    Time: O(n log n)
    Space: O(n)
    '''
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    '''Merge two sorted arrays.'''
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # [11, 12, 22, 25, 34, 64, 90]"""
        },
        {
            "instruction": "Write Python code to find the longest substring without repeating characters",
            "output": """def longest_unique_substring(s):
    '''Find longest substring without repeating characters.
    
    Uses sliding window approach - O(n) time complexity.
    
    Args:
        s: Input string
        
    Returns:
        Tuple of (length, substring)
    '''
    if not s:
        return 0, ""
    
    char_index = {}
    max_length = 0
    max_start = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        
        if end - start + 1 > max_length:
            max_length = end - start + 1
            max_start = start
    
    return max_length, s[max_start:max_start + max_length]

# Examples
print(longest_unique_substring("abcabcbb"))  # (3, "abc")
print(longest_unique_substring("bbbbb"))    # (1, "b")
print(longest_unique_substring("pwwkew"))    # (3, "wke")"""
        },
    ]
    
    samples = []
    for i in range(count):
        problem = problems[i % len(problems)]
        samples.append(format_sample(problem["instruction"], problem["output"]))
    
    return samples

def setup_model():
    print("\n[LOADING BASE MODEL]")
    model_path = "C:/Dev/Projects/BlitzKode/models/qwen1.5b"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA config
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
    print("BLITZKODE V3 - MAXIMUM TRAINING")
    print("="*60)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    dataset = load_datasets()
    model, tokenizer = setup_model()
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  # Shorter for faster training
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Optimized training args for speed
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        warmup_steps=50,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="none",
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    print("\n[TRAINING STARTED]")
    trainer.train()
    
    print("\n[SAVING MODEL]")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    print(f"\n[DONE] Model saved to: {OUTPUT_DIR / 'final'}")

if __name__ == "__main__":
    main()
