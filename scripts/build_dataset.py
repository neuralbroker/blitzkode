#!/usr/bin/env python3
"""Stage 1: Build Training Dataset"""

import json
from pathlib import Path

SAMPLES = [
    {
        "prompt": "Write a Python function two_sum(nums, target) that returns indices of two numbers that add up to target.",
        "response": """Use hash map for O(n) solution.
```python
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
```"""
    },
    {
        "prompt": "Implement a Python LRU Cache with O(1) get and put.",
        "response": """Use hashmap + doubly linked list.
```python
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
```"""
    },
    {
        "prompt": "Write a JavaScript debounce function.",
        "response": """```javascript
function debounce(fn, wait) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), wait);
    };
}
```"""
    }
]

Path("datasets/raw").mkdir(parents=True, exist_ok=True)
with open("datasets/raw/blitzkode_sft_v1.json", "w") as f:
    json.dump(SAMPLES, f, indent=2)

print(f"Created {len(SAMPLES)} samples")
print("NEXT: python scripts/train_sft.py")
