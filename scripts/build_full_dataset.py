#!/usr/bin/env python3
"""
Comprehensive Dataset Builder
Gathers training data from HuggingFace, generates coding problems, etc.
"""

import json
from bisect import bisect_left
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("datasets/raw")
OUTPUT_FILE = OUTPUT_DIR / "blitzkode_full_training.json"

def load_huggingface_datasets():
    """Try to load datasets from HuggingFace"""
    datasets = []
    
    try:
        from datasets import load_dataset
        print("[HF] Loading datasets...")
        
        # Try different datasets
        dataset_names = [
            "openai/gsm8k",
            "meta-math/MetaMathQA",
            "meta-math/MathInstruct",
            "技术桃子/gsm8k-grade-school-math",
        ]
        
        for name in dataset_names:
            try:
                print(f"  Trying {name}...")
                ds = load_dataset(name, split="train[:100]")
                print(f"  Loaded {len(ds)} from {name}")
                for item in ds:
                    datasets.append({
                        "prompt": str(item.get("question", item.get("instruction", "")))[:500],
                        "response": str(item.get("answer", item.get("response", item.get("output", ""))))[:1000]
                    })
            except Exception as e:
                print(f"  Failed: {e}")
                
    except ImportError:
        print("[HF] datasets library not available")
    
    return datasets

def generate_comprehensive_problems():
    """Generate 500+ comprehensive coding problems"""
    
    problems = []
    
    # ==================== ARRAYS ====================
    array_problems = [
        {"prompt": "Two Sum: Given array nums and target, return indices of two numbers that add to target", 
         "response": "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target-n], i]\n        seen[n] = i\n    return []\n```\nTime O(n), Space O(n)"},
        
        {"prompt": "Maximum Subarray: Find contiguous subarray with largest sum (Kadane's algorithm)",
         "response": "```python\ndef max_subarray(nums):\n    max_sum = curr = nums[0]\n    for n in nums[1:]:\n        curr = max(n, curr + n)\n        max_sum = max(max_sum, curr)\n    return max_sum\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Product of Array Except Self: Return array where each element is product of all other elements",
         "response": "```python\ndef product_except_self(nums):\n    n = len(nums)\n    result = [1] * n\n    prefix = 1\n    for i in range(n):\n        result[i] = prefix\n        prefix *= nums[i]\n    suffix = 1\n    for i in range(n-1, -1, -1):\n        result[i] *= suffix\n        suffix *= nums[i]\n    return result\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Merge Intervals: Merge all overlapping intervals",
         "response": "```python\ndef merge(intervals):\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        if merged[-1][1] >= start:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```\nTime O(n log n), Space O(n)"},
        
        {"prompt": "Rotate Array: Rotate array to right by k steps",
         "response": "```python\ndef rotate(nums, k):\n    k = k % len(nums)\n    nums[:] = nums[-k:] + nums[:-k]\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Find Duplicate: Find the duplicate number in array where all elements are in [1,n]",
         "response": "```python\ndef find_duplicate(nums):\n    slow = fast = nums[0]\n    while True:\n        slow = nums[slow]\n        fast = nums[nums[fast]]\n        if slow == fast:\n            break\n    slow = nums[0]\n    while slow != fast:\n        slow = nums[slow]\n        fast = nums[fast]\n    return slow\n```\nTime O(n), Space O(1) - Floyd's Tortoise"},
        
        {"prompt": "Container With Most Water: Find max water container between vertical lines",
         "response": "```python\ndef max_area(height):\n    l, r, max_water = 0, len(height)-1, 0\n    while l < r:\n        water = min(height[l], height[r]) * (r - l)\n        max_water = max(max_water, water)\n        if height[l] < height[r]:\n            l += 1\n        else:\n            r -= 1\n    return max_water\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "3Sum: Find all unique triplets that sum to zero",
         "response": "```python\ndef three_sum(nums):\n    res = []\n    nums.sort()\n    for i in range(len(nums)-2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        l, r = i+1, len(nums)-1\n        while l < r:\n            s = nums[i]+nums[l]+nums[r]\n            if s == 0:\n                res.append([nums[i],nums[l],nums[r]])\n                l += 1; r -= 1\n                while l < r and nums[l] == nums[l-1]: l += 1\n            elif s < 0: l += 1\n            else: r -= 1\n    return res\n```\nTime O(n^2), Space O(n)"},
        
        {"prompt": "Dutch National Flag: Sort 0s, 1s, 2s in-place",
         "response": "```python\ndef sort_colors(nums):\n    l, r = 0, len(nums)-1\n    i = 0\n    while i <= r:\n        if nums[i] == 0:\n            nums[i], nums[l] = nums[l], nums[i]\n            l += 1; i += 1\n        elif nums[i] == 2:\n            nums[i], nums[r] = nums[r], nums[i]\n            r -= 1\n        else:\n            i += 1\n    return nums\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Subarray Sum Equals K: Find number of continuous subarrays that sum to k",
         "response": "```python\ndef subarray_sum(nums, k):\n    count = 0\n    cumsum = 0\n    seen = {0: 1}\n    for n in nums:\n        cumsum += n\n        count += seen.get(cumsum - k, 0)\n        seen[cumsum] = seen.get(cumsum, 0) + 1\n    return count\n```\nTime O(n), Space O(n)"},
    ]
    problems.extend(array_problems)
    
    # ==================== STRINGS ====================
    string_problems = [
        {"prompt": "Valid Palindrome: Check if string is palindrome ignoring non-alphanumeric",
         "response": "```python\ndef is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n```\nTime O(n), Space O(n)"},
        
        {"prompt": "Longest Substring Without Repeating: Find length of longest unique char substring",
         "response": "```python\ndef length_of_longest_substring(s):\n    char_index = {}\n    max_len = start = 0\n    for i, c in enumerate(s):\n        if c in char_index and char_index[c] >= start:\n            start = char_index[c] + 1\n        char_index[c] = i\n        max_len = max(max_len, i - start + 1)\n    return max_len\n```\nTime O(n), Space O(min(n, alphabet))"},
        
        {"prompt": "String to Integer (atoi): Convert string to 32-bit signed integer",
         "response": "```python\ndef my_atoi(s):\n    s = s.strip()\n    if not s: return 0\n    sign = -1 if s[0] == '-' else 1\n    if s[0] in '+-': s = s[1:]\n    num = 0\n    for c in s:\n        if not c.isdigit(): break\n        num = num * 10 + int(c)\n    return max(-2**31, min(2**31-1, sign * num))\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Implement strStr: Return index of first occurrence of needle in haystack",
         "response": "```python\ndef str_str(haystack, needle):\n    return haystack.find(needle)\n```\nOr KMP: Time O(n+m), Space O(m)"},
        
        {"prompt": "Longest Common Prefix: Find longest common prefix among strings",
         "response": "```python\ndef longest_common_prefix(strs):\n    if not strs: return ''\n    for i, c in enumerate(strs[0]):\n        for s in strs[1:]:\n            if i >= len(s) or s[i] != c:\n                return strs[0][:i]\n    return strs[0]\n```\nTime O(n*m), Space O(1)"},
        
        {"prompt": "Valid Parentheses: Check if parentheses are balanced",
         "response": "```python\ndef is_valid(s):\n    stack = []\n    paren = {')':'(', ']':'[', '}':'{'}\n    for c in s:\n        if c in paren:\n            if not stack or stack.pop() != paren[c]:\n                return False\n        else:\n            stack.append(c)\n    return not stack\n```\nTime O(n), Space O(n)"},
        
        {"prompt": "Word Break: Check if string can be segmented into dictionary words",
         "response": "```python\ndef word_break(s, word_dict):\n    dp = [False] * (len(s)+1)\n    dp[0] = True\n    for i in range(1, len(s)+1):\n        for j in range(i):\n            if dp[j] and s[j:i] in word_dict:\n                dp[i] = True\n                break\n    return dp[len(s)]\n```\nTime O(n^2), Space O(n)"},
        
        {"prompt": "Count and Say: Generate nth term of count-and-say sequence",
         "response": "```python\ndef count_and_say(n):\n    result = '1'\n    for _ in range(n-1):\n        new = ''\n        i = 0\n        while i < len(result):\n            count = 1\n            while i+1 < len(result) and result[i] == result[i+1]:\n                count += 1; i += 1\n            new += str(count) + result[i]\n            i += 1\n        result = new\n    return result\n```\nTime O(n*2^n), Space O(2^n)"},
    ]
    problems.extend(string_problems)
    
    # ==================== LINKED LISTS ====================
    linked_problems = [
        {"prompt": "Reverse Linked List: Reverse singly linked list",
         "response": "```python\ndef reverse_list(head):\n    prev = None\n    while head:\n        next_node = head.next\n        head.next = prev\n        prev = head\n        head = next_node\n    return prev\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Merge Two Sorted Lists: Merge two sorted linked lists",
         "response": "```python\ndef merge_two_lists(l1, l2):\n    dummy = cur = ListNode()\n    while l1 and l2:\n        if l1.val < l2.val:\n            cur.next = l1\n            l1 = l1.next\n        else:\n            cur.next = l2\n            l2 = l2.next\n        cur = cur.next\n    cur.next = l1 or l2\n    return dummy.next\n```\nTime O(n+m), Space O(1)"},
        
        {"prompt": "Linked List Cycle: Detect if linked list has a cycle",
         "response": "```python\ndef has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False\n```\nTime O(n), Space O(1) - Floyd's cycle detection"},
        
        {"prompt": "Remove Nth Node From End: Remove nth node from end of list",
         "response": "```python\ndef remove_nth_from_end(head, n):\n    dummy = ListNode(0, head)\n    slow = fast = dummy\n    for _ in range(n+1):\n        fast = fast.next\n    while fast:\n        slow = slow.next\n        fast = fast.next\n    slow.next = slow.next.next\n    return dummy.next\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Add Two Numbers: Add two numbers represented by linked lists",
         "response": "```python\ndef add_two_numbers(l1, l2):\n    dummy = cur = ListNode()\n    carry = 0\n    while l1 or l2 or carry:\n        s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry\n        carry = s // 10\n        cur.next = ListNode(s % 10)\n        cur = cur.next\n        l1 = l1.next if l1 else None\n        l2 = l2.next if l2 else None\n    return dummy.next\n```\nTime O(max(m,n)), Space O(1)"},
    ]
    problems.extend(linked_problems)
    
    # ==================== TREES ====================
    tree_problems = [
        {"prompt": "Binary Tree Inorder Traversal: Return inorder traversal of BST",
         "response": "```python\ndef inorder(root, res=[]):\n    if root:\n        inorder(root.left, res)\n        res.append(root.val)\n        inorder(root.right, res)\n    return res\n```\nOr iterative: Time O(n), Space O(h)"},
        
        {"prompt": "Maximum Depth of Binary Tree: Find deepest node depth",
         "response": "```python\ndef max_depth(root):\n    if not root: return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))\n```\nTime O(n), Space O(h)"},
        
        {"prompt": "Validate Binary Search Tree: Check if tree is valid BST",
         "response": "```python\ndef is_valid_bst(root, float('-inf'), float('inf')):\n    if not root: return True\n    if root.val <= min_val or root.val >= max_val: return False\n    return is_valid_bst(root.left, min_val, root.val) and \\\n           is_valid_bst(root.right, root.val, max_val)\n```\nTime O(n), Space O(h)"},
        
        {"prompt": "Symmetric Tree: Check if tree is mirror of itself",
         "response": "```python\ndef is_symmetric(root):\n    def helper(l, r):\n        if not l or not r: return l == r\n        return l.val == r.val and helper(l.left, r.right) and helper(l.right, r.left)\n    return helper(root, root)\n```\nTime O(n), Space O(h)"},
        
        {"prompt": "Construct Binary Tree from Preorder and Inorder",
         "response": "```python\ndef build_tree(preorder, inorder):\n    if not preorder: return None\n    root = TreeNode(preorder[0])\n    idx = inorder.index(preorder[0])\n    root.left = build_tree(preorder[1:idx+1], inorder[:idx])\n    root.right = build_tree(preorder[idx+1:], inorder[idx+1:])\n    return root\n```\nTime O(n), Space O(n)"},
        
        {"prompt": "Lowest Common Ancestor of BST",
         "response": "```python\ndef lowest_common_ancestor(root, p, q):\n    while root:\n        if p.val < root.val and q.val < root.val:\n            root = root.left\n        elif p.val > root.val and q.val > root.val:\n            root = root.right\n        else:\n            return root\n    return None\n```\nTime O(h), Space O(1)"},
        
        {"prompt": "Binary Tree Level Order Traversal: BFS traversal",
         "response": "```python\nfrom collections import deque\ndef level_order(root):\n    if not root: return []\n    result, queue = [], deque([root])\n    while queue:\n        level = []\n        for _ in range(len(queue)):\n            node = queue.popleft()\n            level.append(node.val)\n            if node.left: queue.append(node.left)\n            if node.right: queue.append(node.right)\n        result.append(level)\n    return result\n```\nTime O(n), Space O(w)"},
    ]
    problems.extend(tree_problems)
    
    # ==================== DYNAMIC PROGRAMMING ====================
    dp_problems = [
        {"prompt": "Climbing Stairs: Count ways to climb n stairs (1 or 2 steps)",
         "response": "```python\ndef climb_stairs(n):\n    if n <= 2: return n\n    dp = [0] * (n+1)\n    dp[1], dp[2] = 1, 2\n    for i in range(3, n+1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]\n```\nTime O(n), Space O(1) - can optimize to 2 vars"},
        
        {"prompt": "Coin Change: Minimum coins to make amount",
         "response": "```python\ndef coin_change(coins, amount):\n    dp = [float('inf')] * (amount+1)\n    dp[0] = 0\n    for c in coins:\n        for i in range(c, amount+1):\n            dp[i] = min(dp[i], dp[i-c]+1)\n    return dp[amount] if dp[amount] != float('inf') else -1\n```\nTime O(n*amount), Space O(amount)"},
        
        {"prompt": "Longest Increasing Subsequence: Find LIS length",
         "response": "```python\ndef length_of_lis(nums):\n    tails = []\n    for n in nums:\n        i = bisect_left(tails, n)\n        if i == len(tails): tails.append(n)\n        else: tails[i] = n\n    return len(tails)\n```\nTime O(n log n), Space O(n)"},
        
        {"prompt": "Word Break: Can string be segmented into dictionary words",
         "response": "```python\ndef word_break(s, word_dict):\n    dp = [False] * (len(s)+1)\n    dp[0] = True\n    for i in range(1, len(s)+1):\n        for j in range(i):\n            if dp[j] and s[j:i] in word_dict:\n                dp[i] = True\n                break\n    return dp[len(s)]\n```\nTime O(n^2), Space O(n)"},
        
        {"prompt": "House Robber: Max amount without robbing adjacent houses",
         "response": "```python\ndef rob(nums):\n    prev = curr = 0\n    for n in nums:\n        prev, curr = curr, max(curr, prev + n)\n    return curr\n```\nTime O(n), Space O(1)"},
        
        {"prompt": "Edit Distance: Minimum edits to transform word1 to word2",
         "response": "```python\ndef min_distance(word1, word2):\n    m, n = len(word1), len(word2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(m+1): dp[i][0] = i\n    for j in range(n+1): dp[0][j] = j\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if word1[i-1] == word2[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    return dp[m][n]\n```\nTime O(m*n), Space O(m*n)"},
        
        {"prompt": "Longest Common Subsequence: Find LCS length",
         "response": "```python\ndef lcs(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if text1[i-1] == text2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]\n```\nTime O(m*n), Space O(m*n)"},
        
        {"prompt": "Decode Ways: Count ways to decode string",
         "response": "```python\ndef num_decodings(s):\n    if not s or s[0] == '0': return 0\n    dp = [0] * (len(s)+1)\n    dp[0] = dp[1] = 1\n    for i in range(2, len(s)+1):\n        if s[i-1] != '0': dp[i] += dp[i-1]\n        if 10 <= int(s[i-2:i]) <= 26: dp[i] += dp[i-2]\n    return dp[len(s)]\n```\nTime O(n), Space O(1)"},
    ]
    problems.extend(dp_problems)
    
    # ==================== GRAPHS ====================
    graph_problems = [
        {"prompt": "Number of Islands: Count islands in 2D grid",
         "response": "```python\ndef num_islands(grid):\n    def dfs(i, j):\n        grid[i][j] = '0'\n        for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:\n            ni, nj = i+di, j+dj\n            if 0<=ni<m and 0<=nj<n and grid[ni][nj]=='1':\n                dfs(ni, nj)\n    count = 0\n    m, n = len(grid), len(grid[0])\n    for i in range(m):\n        for j in range(n):\n            if grid[i][j] == '1':\n                dfs(i, j)\n                count += 1\n    return count\n```\nTime O(m*n), Space O(m*n)"},
        
        {"prompt": "Clone Graph: Deep copy undirected graph",
         "response": "```python\ndef clone_graph(node):\n    if not node: return None\n    visited = {}\n    def dfs(n):\n        if n in visited: return visited[n]\n        visited[n] = Node(n.val, [])\n        for neighbor in n.neighbors:\n            visited[n].neighbors.append(dfs(neighbor))\n        return visited[n]\n    return dfs(node)\n```\nTime O(V+E), Space O(V)"},
        
        {"prompt": "Course Schedule: Check if can finish all courses",
         "response": "```python\ndef can_finish(numCourses, prerequisites):\n    graph = [[] for _ in range(numCourses)]\n    visited = [0] * numCourses\n    for u, v in prerequisites:\n        graph[v].append(u)\n    def dfs(i):\n        if visited[i] == 1: return False\n        if visited[i] == 2: return True\n        visited[i] = 1\n        for nei in graph[i]:\n            if not dfs(nei): return False\n        visited[i] = 2\n        return True\n    return all(dfs(i) for i in range(numCourses))\n```\nTime O(V+E), Space O(V)"},
        
        {"prompt": "Pacific Atlantic Water Flow: Find cells that can flow to both oceans",
         "response": "```python\ndef pacific_atlantic(heights):\n    if not heights: return []\n    m, n = len(heights), len(heights[0])\n    pacific = [[False]*n for _ in range(m)]\n    atlantic = [[False]*n for _ in range(m)]\n    def dfs(r, c, visited, prev):\n        if r<0 or c<0 or r>=m or c>=n: return\n        if visited[r][c] or heights[r][c] < prev: return\n        visited[r][c] = True\n        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:\n            dfs(r+dr, c+dc, visited, heights[r][c])\n    for i in range(m): dfs(i, 0, pacific, 0); dfs(i, n-1, atlantic, 0)\n    for j in range(n): dfs(0, j, pacific, 0); dfs(m-1, j, atlantic, 0)\n    return [[i,j] for i in range(m) for j in range(n) if pacific[i][j] and atlantic[i][j]]\n```\nTime O(m*n), Space O(m*n)"},
    ]
    problems.extend(graph_problems)
    
    # ==================== SORTING & SEARCHING ====================
    search_problems = [
        {"prompt": "Binary Search: Find target in sorted array",
         "response": "```python\ndef binary_search(arr, target):\n    l, r = 0, len(arr)-1\n    while l <= r:\n        mid = (l + r) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: l = mid + 1\n        else: r = mid - 1\n    return -1\n```\nTime O(log n), Space O(1)"},
        
        {"prompt": "Search in Rotated Array: Search in rotated sorted array",
         "response": "```python\ndef search(nums, target):\n    l, r = 0, len(nums)-1\n    while l <= r:\n        mid = (l + r) // 2\n        if nums[mid] == target: return mid\n        if nums[l] <= nums[mid]:\n            if nums[l] <= target < nums[mid]: r = mid - 1\n            else: l = mid + 1\n        else:\n            if nums[mid] < target <= nums[r]: l = mid + 1\n            else: r = mid - 1\n    return -1\n```\nTime O(log n), Space O(1)"},
        
        {"prompt": "Merge Sorted Array: Merge two sorted arrays in-place",
         "response": "```python\ndef merge(nums1, m, nums2, n):\n    i, j, k = m-1, n-1, m+n-1\n    while i >= 0 and j >= 0:\n        if nums1[i] > nums2[j]:\n            nums1[k] = nums1[i]; i -= 1\n        else:\n            nums1[k] = nums2[j]; j -= 1\n        k -= 1\n    while j >= 0:\n        nums1[k] = nums2[j]; j -= 1; k -= 1\n```\nTime O(m+n), Space O(1)"},
    ]
    problems.extend(search_problems)
    
    # ==================== RECURSION & BACKTRACKING ====================
    backtrack_problems = [
        {"prompt": "Generate all permutations of array",
         "response": "```python\ndef permute(nums):\n    res = []\n    def backtrack(path, used):\n        if len(path) == len(nums):\n            res.append(path[:])\n            return\n        for i, n in enumerate(nums):\n            if i in used: continue\n            path.append(n)\n            used.add(i)\n            backtrack(path, used)\n            path.pop()\n            used.remove(i)\n    backtrack([], set())\n    return res\n```\nTime O(n!*n), Space O(n)"},
        
        {"prompt": "Generate all subsets (Power Set)",
         "response": "```python\ndef subsets(nums):\n    res = []\n    def backtrack(i, path):\n        res.append(path[:])\n        for j in range(i, len(nums)):\n            path.append(nums[j])\n            backtrack(j+1, path)\n            path.pop()\n    backtrack(0, [])\n    return res\n```\nTime O(n*2^n), Space O(n)"},
        
        {"prompt": "N-Queens: Place N queens on NxN board",
          "response": "```python\ndef solve_n_queens(n):\n    res = []\n    cols = set()\n    pos_diag = set()\n    neg_diag = set()\n    board = [['.' for _ in range(n)] for _ in range(n)]\n    def backtrack(row):\n        if row == n:\n            res.append([''.join(row) for row in board])\n            return\n        for col in range(n):\n            if col in cols or row-col in pos_diag or row+col in neg_diag:\n                continue\n            board[row][col] = 'Q'\n            cols.add(col); pos_diag.add(row-col); neg_diag.add(row+col)\n            backtrack(row+1)\n            board[row][col] = '.'\n            cols.remove(col); pos_diag.remove(row-col); neg_diag.remove(row+col)\n    backtrack(0)\n    return res\n```\nTime O(N!), Space O(N)"},
        
        {"prompt": "Combination Sum: Find all combinations that sum to target",
         "response": "```python\ndef combination_sum(candidates, target):\n    res = []\n    candidates.sort()\n    def backtrack(i, path, remain):\n        if remain == 0: res.append(path[:])\n        if remain < 0: return\n        for j in range(i, len(candidates)):\n            path.append(candidates[j])\n            backtrack(j, path, remain-candidates[j])\n            path.pop()\n    backtrack(0, [], target)\n    return res\n```\nTime O(2^n), Space O(n)"},
    ]
    problems.extend(backtrack_problems)
    
    # ==================== DESIGN ====================
    design_problems = [
        {"prompt": "Implement LRU Cache with O(1) get and put",
         "response": "```python\nfrom collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key not in self.cache: return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache: self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```\nTime O(1), Space O(capacity)"},
        
        {"prompt": "Implement Trie (Prefix Tree)",
         "response": "```python\nclass TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\nclass Trie:\n    def __init__(self): self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children: node.children[c] = TrieNode()\n            node = node.children[c]\n        node.is_end = True\n    def search(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children: return False\n            node = node.children[c]\n        return node.is_end\n    def startsWith(self, prefix):\n        node = self.root\n        for c in prefix:\n            if c not in node.children: return False\n            node = node.children[c]\n        return True\n```\nTime O(m), Space O(m) per operation"},
        
        {"prompt": "Min Stack: Stack that supports min in O(1)",
         "response": "```python\nclass MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n    def push(self, val):\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    def pop(self):\n        val = self.stack.pop()\n        if val == self.min_stack[-1]: self.min_stack.pop()\n        return val\n    def top(self): return self.stack[-1]\n    def getMin(self): return self.min_stack[-1]\n```\nTime O(1), Space O(n)"},
        
        {"prompt": "Implement Rate Limiter (Token Bucket)",
         "response": "```python\nimport time\nclass RateLimiter:\n    def __init__(self, rate, capacity):\n        self.rate = rate\n        self.capacity = capacity\n        self.tokens = capacity\n        self.last_time = time.time()\n    def allow(self):\n        now = time.time()\n        self.tokens = min(self.capacity, self.tokens + (now-self.last_time)*self.rate)\n        self.last_time = now\n        if self.tokens >= 1:\n            self.tokens -= 1\n            return True\n        return False\n```\nTime O(1), Space O(1)"},
    ]
    problems.extend(design_problems)
    
    # ==================== JAVASCRIPT ====================
    js_problems = [
        {"prompt": "JavaScript: Debounce function",
         "response": "```javascript\nfunction debounce(func, wait) {\n  let timeout;\n  return function(...args) {\n    clearTimeout(timeout);\n    timeout = setTimeout(() => func.apply(this, args), wait);\n  };\n}\n```"},
        
        {"prompt": "JavaScript: Throttle function",
         "response": "```javascript\nfunction throttle(func, limit) {\n  let inThrottle;\n  return function(...args) {\n    if (!inThrottle) {\n      func.apply(this, args);\n      inThrottle = true;\n      setTimeout(() => inThrottle = false, limit);\n    }\n  };\n}\n```"},
        
        {"prompt": "JavaScript: Deep clone object",
         "response": "```javascript\nfunction deepClone(obj, visited = new WeakMap()) {\n  if (obj === null || typeof obj !== 'object') return obj;\n  if (visited.has(obj)) return visited.get(obj);\n  const clone = Array.isArray(obj) ? [] : {};\n  visited.set(obj, clone);\n  for (const key in obj) {\n    if (obj.hasOwnProperty(key)) {\n      clone[key] = deepClone(obj[key], visited);\n    }\n  }\n  return clone;\n}\n```"},
        
        {"prompt": "JavaScript: Promise.all with timeout",
         "response": "```javascript\nfunction promiseAllWithTimeout(promises, timeout) {\n  return Promise.all(\n    promises.map(p =>\n      Promise.race([\n        p,\n        new Promise((_, reject) => \n          setTimeout(() => reject(new Error('Timeout')), timeout)\n        )\n      ])\n    )\n  );\n}\n```"},
        
        {"prompt": "JavaScript: Event Emitter implementation",
         "response": "```javascript\nclass EventEmitter {\n  constructor() { this.events = {}; }\n  on(event, fn) {\n    if (!this.events[event]) this.events[event] = [];\n    this.events[event].push(fn);\n  }\n  emit(event, ...args) {\n    if (this.events[event]) \n      this.events[event].forEach(fn => fn(...args));\n  }\n  off(event, fn) {\n    if (this.events[event]) \n      this.events[event] = this.events[event].filter(f => f !== fn);\n  }\n}\n```"},
        
        {"prompt": "JavaScript: Currying function",
         "response": "```javascript\nfunction curry(fn) {\n  return function curried(...args) {\n    if (args.length >= fn.length) {\n      return fn.apply(this, args);\n    }\n    return function(...args2) {\n      return curried.apply(this, args.concat(args2));\n    };\n  };\n}\nconst add = curry((a, b, c) => a + b + c);\nadd(1)(2)(3); // 6\n```"},
    ]
    problems.extend(js_problems)
    
    # ==================== JAVA ====================
    java_problems = [
        {"prompt": "Java: Thread-safe Singleton (Double-checked locking)",
         "response": "```java\npublic class Singleton {\n    private static volatile Singleton instance;\n    private Singleton() {}\n    public static Singleton getInstance() {\n        if (instance == null) {\n            synchronized (Singleton.class) {\n                if (instance == null) {\n                    instance = new Singleton();\n                }\n            }\n        }\n        return instance;\n    }\n}\n```"},
        
        {"prompt": "Java: Producer-Consumer with wait/notify",
         "response": "```java\npublic class ProducerConsumer {\n    private Queue<Integer> buffer = new LinkedList<>();\n    private int MAX_SIZE = 10;\n    \n    public synchronized void produce(int value) throws InterruptedException {\n        while (buffer.size() >= MAX_SIZE) wait();\n        buffer.add(value);\n        notifyAll();\n    }\n    \n    public synchronized int consume() throws InterruptedException {\n        while (buffer.isEmpty()) wait();\n        notifyAll();\n        return buffer.poll();\n    }\n}\n```"},
        
        {"prompt": "Java: Builder Pattern for User class",
         "response": "```java\npublic class User {\n    private String name, email;\n    private int age;\n    \n    private User(Builder b) {\n        this.name = b.name; this.email = b.email; this.age = b.age;\n    }\n    \n    public static class Builder {\n        private String name, email;\n        private int age = 0;\n        public Builder name(String name) { this.name = name; return this; }\n        public Builder email(String email) { this.email = email; return this; }\n        public Builder age(int age) { this.age = age; return this; }\n        public User build() { return new User(this); }\n    }\n}\n// Usage: User u = new User.Builder().name(\"John\").age(25).build();\n```"},
        
        {"prompt": "Java: Generic Stack implementation",
         "response": "```java\npublic class Stack<T> {\n    private Object[] arr;\n    private int top = -1, capacity;\n    \n    public Stack(int capacity) {\n        this.capacity = capacity;\n        arr = new Object[capacity];\n    }\n    \n    public void push(T item) {\n        if (top == capacity - 1) throw new StackOverflowError();\n        arr[++top] = item;\n    }\n    \n    @SuppressWarnings(\"unchecked\")\n    public T pop() {\n        if (top == -1) throw new EmptyStackException();\n        return (T) arr[top--];\n    }\n    \n    public T peek() {\n        if (top == -1) throw new EmptyStackException();\n        return (T) arr[top];\n    }\n}\n```"},
        
        {"prompt": "Java: Observer Pattern",
         "response": "```java\ninterface Observer { void update(String msg); }\nclass Subject {\n    private List<Observer> observers = new ArrayList<>();\n    public void attach(Observer o) { observers.add(o); }\n    public void detach(Observer o) { observers.remove(o); }\n    public void notifyAll(String msg) {\n        observers.forEach(o -> o.update(msg));\n    }\n}\n```"},
    ]
    problems.extend(java_problems)
    
    # ==================== ALGORITHMS & CONCEPTS ====================
    concept_problems = [
        {"prompt": "Explain Big O, Big Omega, Big Theta notation",
         "response": "Big O: Upper bound (worst case)\nBig Omega: Lower bound (best case)\nBig Theta: Tight bound (average)\n\nExamples:\n- O(1): Array access\n- O(log n): Binary search\n- O(n): Linear search\n- O(n log n): Merge sort\n- O(n^2): Bubble sort\n- O(2^n): Recursive Fibonacci\n- O(n!): Permutations"},
        
        {"prompt": "Explain and implement QuickSort",
         "response": "```python\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```\nAverage: O(n log n), Worst: O(n^2)\nSpace: O(n)"},
        
        {"prompt": "Explain and implement MergeSort",
         "response": "```python\ndef mergesort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    left = mergesort(arr[:mid])\n    right = mergesort(arr[mid:])\n    return merge(left, right)\n\ndef merge(l, r):\n    result = []\n    while l and r:\n        result.append(l.pop(0) if l[0] <= r[0] else r.pop(0))\n    result.extend(l or r)\n    return result\n```\nTime: O(n log n), Space: O(n)\nStable sort"},
        
        {"prompt": "Explain Hash Table collision resolution",
         "response": "1. Chaining: Store collisions in linked list\n   - Time: O(1) avg, O(n) worst\n2. Open Addressing: Find next empty slot\n   - Linear probing: h(k) + i\n   - Quadratic probing: h(k) + i^2\n   - Double hashing: h1(k) + i*h2(k)\n\nLoad factor = n/m should be < 0.7"},
        
        {"prompt": "Explain BFS vs DFS",
         "response": "BFS (Breadth-First Search):\n- Uses Queue\n- Level by level traversal\n- Finds shortest path in unweighted\n- More memory\n- Good for shortest path\n\nDFS (Depth-First Search):\n- Uses Stack (or recursion)\n- Goes deep first\n- Lower memory for deep graphs\n- Good for path finding, cycles"},
        
        {"prompt": "Explain Dijkstra's algorithm",
         "response": "```python\nimport heapq\ndef dijkstra(graph, start):\n    dist = {v: float('inf') for v in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]: continue\n        for v, w in graph.get(u, []):\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist\n```\nTime: O((V+E) log V)\nSpace: O(V)\nGreedy, doesn't work with negative weights"},
        
        {"prompt": "Explain Union-Find (Disjoint Set)",
         "response": "```python\nclass UnionFind:\n    def __init__(self, n):\n        self.parent = list(range(n))\n        self.rank = [0] * n\n    \n    def find(self, x):\n        if self.parent[x] != x:\n            self.parent[x] = self.find(self.parent[x])\n        return self.parent[x]\n    \n    def union(self, x, y):\n        px, py = self.find(x), self.find(y)\n        if px == py: return\n        if self.rank[px] < self.rank[py]: px, py = py, px\n        self.parent[py] = px\n        if self.rank[px] == self.rank[py]: self.rank[px] += 1\n```\nTime: O(alpha(n)) - almost constant"},
    ]
    problems.extend(concept_problems)
    
    print(f"Generated {len(problems)} comprehensive problems")
    return problems

def main():
    print("=" * 60)
    print("BUILDING COMPREHENSIVE DATASET")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # 1. Generate comprehensive problems
    print("\n[1/3] Generating algorithmic problems...")
    problems = generate_comprehensive_problems()
    all_data.extend(problems)
    print(f"  Total: {len(all_data)}")
    
    # 2. Try HuggingFace datasets
    print("\n[2/3] Loading HuggingFace datasets...")
    hf_data = load_huggingface_datasets()
    all_data.extend(hf_data)
    print(f"  Total: {len(all_data)}")
    
    # 3. Add more diverse problems
    print("\n[3/3] Adding additional problems...")
    
    additional = [
        {"prompt": "Python: Read file and count word frequencies", 
         "response": "```python\nfrom collections import Counter\nwith open('file.txt') as f:\n    words = f.read().split()\nprint(Counter(words).most_common(10))\n```"},
        
        {"prompt": "Python: Async HTTP request with aiohttp",
         "response": "```python\nimport aiohttp, asyncio\nasync def fetch(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.text()\nasyncio.run(fetch('https://example.com'))\n```"},
        
        {"prompt": "Python: Context manager for timing",
         "response": "```python\nimport time\nclass Timer:\n    def __enter__(self): self.start = time.time()\n    def __exit__(self, *args): print(time.time()-self.start)\nwith Timer():\n    # code to time\n    sum(range(1000000))\n```"},
        
        {"prompt": "Python: Decorator for retry logic",
         "response": "```python\nimport time, functools\ndef retry(max_attempts=3, delay=1):\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for _ in range(max_attempts):\n                try: return func(*args, **kwargs)\n                except Exception as e:\n                    if _ == max_attempts-1: raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\n@retry(max_attempts=3)\ndef call_api(): pass\n```"},
        
        {"prompt": "Python: Generate random password",
         "response": "```python\nimport string, random\ndef generate_password(length=16):\n    chars = string.ascii_letters + string.digits + string.punctuation\n    return ''.join(random.choice(chars) for _ in range(length))\n```"},
        
        {"prompt": "Python: Flask REST API endpoint",
         "response": "```python\nfrom flask import Flask, request, jsonify\napp = Flask(__name__)\n\n@app.route('/api/users/<int:user_id>', methods=['GET'])\ndef get_user(user_id):\n    return jsonify({'id': user_id, 'name': 'John'})\n\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.json\n    return jsonify({'id': 1, **data}), 201\n```"},
        
        {"prompt": "Python: SQLAlchemy model definition",
         "response": "```python\nfrom sqlalchemy import Column, Integer, String, ForeignKey\nfrom sqlalchemy.orm import relationship, declarative_base\nBase = declarative_base()\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(100), nullable=False)\n    email = Column(String(100), unique=True)\n    posts = relationship('Post', back_populates='author')\n\nclass Post(Base):\n    __tablename__ = 'posts'\n    id = Column(Integer, primary_key=True)\n    title = Column(String(200))\n    user_id = Column(Integer, ForeignKey('users.id'))\n    author = relationship('User', back_populates='posts')\n```"},
        
        {"prompt": "Python: pytest fixture example",
         "response": "```python\nimport pytest\n\n@pytest.fixture\ndef db_connection():\n    conn = create_connection()\n    yield conn\n    conn.close()\n\ndef test_user_creation(db_connection):\n    user = create_user(db_connection, 'John')\n    assert user.id is not None\n```"},
        
        {"prompt": "Python: Logging configuration",
         "response": "```python\nimport logging\nlogging.basicConfig(\n    level=logging.INFO,\n    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\n    handlers=[\n        logging.FileHandler('app.log'),\n        logging.StreamHandler()\n    ]\n)\nlogger = logging.getLogger(__name__)\nlogger.info('Application started')\n```"},
        
        {"prompt": "Python: Thread pool executor",
         "response": "```python\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef process_item(item):\n    # do work\n    return result\n\nwith ThreadPoolExecutor(max_workers=4) as executor:\n    results = list(executor.map(process_item, items))\n```"},
        
        {"prompt": "Python: Unit test with unittest",
         "response": "```python\nimport unittest\n\nclass TestMath(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(1+1, 2)\n    \n    def test_divide(self):\n        with self.assertRaises(ZeroDivisionError):\n            1/0\n\nif __name__ == '__main__':\n    unittest.main()\n```"},
        
        {"prompt": "Python: Data class with validation",
         "response": "```python\nfrom dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass User:\n    name: str\n    email: str\n    age: int = 0\n    tags: List[str] = field(default_factory=list)\n    \n    def __post_init__(self):\n        if '@' not in self.email:\n            raise ValueError('Invalid email')\n```"},
        
        {"prompt": "Python: FastAPI endpoint with Pydantic",
         "response": "```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel, EmailStr\n\napp = FastAPI()\n\nclass UserCreate(BaseModel):\n    name: str\n    email: EmailStr\n    age: int = None\n\n@app.post('/users', response_model=UserCreate)\ndef create_user(user: UserCreate):\n    return user\n```"},
        
        {"prompt": "Python: Celery task definition",
         "response": "```python\nfrom celery import Celery\napp = Celery('tasks')\n\n@app.task\ndef process_data(data):\n    # expensive computation\n    result = heavy_computation(data)\n    return result\n\n# Call asynchronously\nresult = process_data.delay(data)\nprint(result.get(timeout=10))\n```"},
        
        {"prompt": "Python: Generator for large file processing",
         "response": "```python\ndef process_large_file(filepath):\n    with open(filepath) as f:\n        for line in f:\n            yield process_line(line)\n\nfor result in process_large_file('huge.txt'):\n    save_to_db(result)\n```\nMemory efficient for large files"},
        
        {"prompt": "Python: Memoization decorator",
         "response": "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\n\n# Or custom:\ndef memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n```"},
        
        {"prompt": "Python: Property-based testing with hypothesis",
         "response": "```python\nfrom hypothesis import given, strategies as st\n\n@given(st.lists(st.integers(min_value=1)))\ndef test_sorting(lst):\n    result = sorted(lst)\n    assert len(result) == len(lst)\n    assert all(result[i] <= result[i+1] for i in range(len(result)-1))\n```"},
        
        {"prompt": "Python: NamedTuple for immutable data",
         "response": "```python\nfrom collections import namedtuple\n\nPoint = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)\nprint(p.x, p.y)  # 1 2\nprint(p._fields)  # ('x', 'y')\n```"},
        
        {"prompt": "Python: Enum for constants",
         "response": "```python\nfrom enum import Enum, auto\n\nclass Status(Enum):\n    PENDING = auto()\n    PROCESSING = auto()\n    COMPLETED = auto()\n    FAILED = auto()\n\nprint(Status.PENDING.name)  # PENDING\nprint(Status.PENDING.value)  # 1\n```"},
        
        {"prompt": "Python: ABC abstract base class",
         "response": "```python\nfrom abc import ABC, abstractmethod\n\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): pass\n    \n    def describe(self):\n        return f'Area: {self.area()}'\n\nclass Circle(Shape):\n    def __init__(self, r): self.r = r\n    def area(self): return 3.14 * self.r ** 2\n```"},
        
        {"prompt": "Python: HTTP server with http.server",
         "response": "```python\nfrom http.server import HTTPServer, BaseHTTPRequestHandler\nimport json\n\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-Type', 'application/json')\n        self.end_headers()\n        self.wfile.write(json.dumps({'message': 'OK'}).encode())\n\nHTTPServer(('localhost', 8000), Handler).serve_forever()\n```"},
        
        {"prompt": "Python: SQLite connection and queries",
         "response": "```python\nimport sqlite3\n\nconn = sqlite3.connect('database.db')\ncursor = conn.cursor()\n\ncursor.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)''')\ncursor.execute('INSERT INTO users (name) VALUES (?)', ('John',))\nconn.commit()\n\nfor row in cursor.execute('SELECT * FROM users'):\n    print(row)\n\nconn.close()\n```"},
        
        {"prompt": "Python: XML parsing with ElementTree",
         "response": "```python\nimport xml.etree.ElementTree as ET\n\ntree = ET.parse('data.xml')\nroot = tree.getroot()\n\nfor child in root.findall('.//item'):\n    print(child.get('id'), child.text)\n\n# Or create:\nroot = ET.Element('root')\nchild = ET.SubElement(root, 'item', id='1')\nchild.text = 'value'\nET.dump(root)\n```"},
        
        {"prompt": "Python: JSON handling",
         "response": "```python\nimport json\n\n# Load\nwith open('data.json') as f:\n    data = json.load(f)\n\n# Parse string\nobj = json.loads('{\"key\": \"value\"}')\n\n# Save\nwith open('out.json', 'w') as f:\n    json.dump(data, f, indent=2)\n\n# Custom encoder\nclass CustomEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, datetime):\n            return obj.isoformat()\n```"},
        
        {"prompt": "Python: dataclasses.asdict()",
         "response": "```python\nfrom dataclasses import dataclass, asdict\n\n@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 8080\n    debug: bool = False\n\nconfig = Config()\ndict_config = asdict(config)\nprint(dict_config)  # {'host': 'localhost', 'port': 8080, 'debug': False}\n```"},
        
        {"prompt": "Python: Enum with methods",
         "response": "```python\nfrom enum import Enum\n\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    \n    def is_primary(self):\n        return self in (Color.RED, Color.GREEN, Color.BLUE)\n\nprint(Color.RED.is_primary())  # True\n```"},
        
        {"prompt": "Python: argparse CLI",
         "response": "```python\nimport argparse\n\nparser = argparse.ArgumentParser(description='My tool')\nparser.add_argument('-n', '--name', required=True, help='Your name')\nparser.add_argument('-v', '--verbose', action='store_true')\nparser.add_argument('-c', '--count', type=int, default=1)\n\nargs = parser.parse_args()\nprint(f'Hello {args.name}' * args.count)\n```\nRun: python script.py -n John -v -c 3"},
        
        {"prompt": "Python: Simple CLI with click",
         "response": "```python\nimport click\n\n@click.command()\n@click.option('--name', default='World', help='Name to greet')\n@click.argument('count', default=1, type=int)\ndef hello(name, count):\n    for _ in range(count):\n        click.echo(f'Hello {name}!')\n\nif __name__ == '__main__':\n    hello()\n```\nRun: python script.py John --count 3"},
    ]
    
    all_data.extend(additional)
    
    print(f"\n[TOTAL] {len(all_data)} samples")
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    
    return len(all_data)

if __name__ == "__main__":
    main()
