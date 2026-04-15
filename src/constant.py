competitive_datasets = ["lcb", "code_contests"]

simple_datasets = ["humanevalplus", "mbppplus", "mbpp", "humaneval"]

import_helper = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"

RESOLVE_SYSTEM = """You are a programming helper. Your task is to write a *Solution Plan* for the given problem.

Rules:
- You are solving the **current problem**, not the full task.
- Use provided sub-problems and their solutions as references if available.
- Do NOT output code. Only reasoning and planning steps.
- If all sub-problems are already solved, integrate their reasoning to form a coherent overall solution."""


PLAN_DESIGN_SYSTEM = """You are a programming helper. Your task is to write a *Solution Plan* for the problem by divide and conquer in multiple turns.

Rules:
Given a programming problem to solve, and current problem, you need to focus on solving the current problem in each turn and attempt to decompose it into sub-problems if it is complex.
For Each turn: 
    - If the current problem is simple, directly solve it in the current turn.
    - If the current problem is complex, divide it into multiple sub-problems.
    Do not include any codes or implementation details in your response.

For Each Turn, Your response must contain:
1. Current Problem: what is the current problem to solve.
2.  - If the current problem is complex to solve, list divided sub-problems.
    - If the current problem is simple and direct, provide a concrete solution to solve it. 
    
If a problem:
    - No further reasoning required — the problem can be solved directly without multi-step logic or intermediate derivations.
    - Operates on local or concrete data — it deals with specific values, strings, or conditions rather than abstract concepts.
    - Independently solvable — it can be completed without relying on other sub-tasks or broader context.
Then, it is considered a **simple problem**. 

## Example:

(Turn 1)
Assistant Output:
### Current Problem:
Compute the sum of all common prime factors of integers a and b.

### Sub-Problems:
1. How to find factors of a number ?
2. Given a number, how to determine whether it is prime or not ?
3. How to find common elements between two lists

(Turn 2)
Assitant Output:
### Current Problem:
How to find factors of a number ?

### Solution:
To find all divisors of a number n, we need to check numbers from 1 to sqrt n.
Because if i is a divisor of n, then n / i is also a divisor. So each divisor pair can be found by checking only up to the square root.
(finding factors of a given number is simple, so we directly solve the problem here.)

(Turn 3)
Assitant Output:
### Current Problem:
Given a number, how to determine whether it is prime or not ?

### Solution:
We can use Trial Division to determine if a factor is prime by checking divisibility from 2 up to the square root of the factor.
(determining whether a number is prime or not is simple, so we directly solve the problem here.)

(Turn 4)
Assitant Output:
### Current Problem:
How to find common elements between two lists

### Solution:
We can convert list into sets and use set intersection to find common elements efficiently.
(find common elements between two lists is simple, so we directly solve the problem here.)"""


CONQUER_SYSTEM = '''You are an AI planning and reasoning assistant. You will receive a **hierarchical plan tree** designed to decompose a programming problem. The tree is formatted as follows:

### Structure Definition
* The root starts with:
`To solve this problem, we can further break it down into more sub-problems:`
* Each sub-problem node is prefixed with:
`- sub-problem X:`
where `X` uses a hierarchical index (e.g. `1`, `1-2`, `2-3-1`…), indicating its dependency relations.
* Each sub-problem node may include:
A *Current Problem* description (the task for this node to solve)
A *Solution* section (the proposed approach for **current** sub-problem)
Or further decomposed child sub-problems

### Semantics of the Plan Tree

* Higher-level sub-problems represent broader tasks.
* Deeper indexed nodes are **supporting sub-problems**, which must be solved first because they enable the parent.
* The goal is to **integrate** all the sub-solutions from leaf nodes upward, forming a complete and logically coherent solution to the root problem.

### Your Task

1. Read and understand the entire plan tree.
2. Ignore the hierarchical numbering in the final write-up (it is only for dependency guidance).
3. Merge all **leaf-node Solution sections** properly into a single, holistic **final solution plan**.
4. Ensure the final solution:

* Preserves the logical order implied by the hierarchy
* Removes redundancy and repeated questions
* Enhances clarity, structure, and reasoning consistency
* Writes in clean and professional technical language
5. The final output should be formatted with:

* A clear step-by-step **plan** or algorithm
* Key justifications or reasoning when needed

### Important rules

* If a node has **only a question but no solution**, use the solutions from its children to answer it.
* If both parent and child provide solutions to the **same** concept, produce a unified improved version.
* Omit auxiliary labels like `Current Problem:` or `Solution:` in the final plan.

---

### Final Expected Output

A **single** well-organized **complete problem-solving plan**, structured logically from start to end, ready to be implemented.
Response starts with **Solution Steps**.

Example Format:

### Solution Steps:
1. (the first step here)..
2. ...'''


fact_sys = '''You previously generated a plan consisting of multiple steps and then produced modular code based on that plan. After Executing the code with sample input, and below are the partial actual console outputs produced by each step.

Your task:
1. Identify and compare **each step's** output against the expected logic described in the original plan.
2. Determine whether each step is correct.
3. Use **0 for incorrect** and **1 for correct**. (If a step does not print anything, infer its correctness from the context, including: Inputs and outputs of surrounding steps and the intended logic of that step in the plan.)

The result of the code execution may only contain the output of **some steps (not all)**. You need to evaluate **each step** of the **solution plan**. 

Return the evaluation strictly in the following **YAML format** (No extra text outside):

```yaml
steps:
  - step 1:
    name: |
      Description of the first step
    score: 1
    reason: |
      Brief explanation why step 1 is correct or incorrect

  - step 2:
    name: |
      Description of the second step
    score: 0
    reason: |
      Brief explanation why step 2 is correct or incorrect
...
  - step k: 
    name: |
      Description of the k step
    score: 1
    reason: |
      Brief explanation why step k is correct or incorrect
```

Attention: Ensure that **the k of the last step** is equal to **the total number** of solution steps.'''


call_based_coding_sys = f'''You are an expert Python programmer. Your task is to generate Python code to solve the given problem based on the following **solution steps**. Follow these strict requirements:

1. **Modular design**: Each step in the solution plan must be implemented as a **separate class member function** within the `Solution` class.
2. **Step execution**: After defining all functions, you must use **the method defined in the question** that calls each step function **in order**, following the logic of the provided solution steps.
3. **Input/output visibility**: Every step function must **print its input(s) and output(s)** clearly for debugging and transparency.
4. **Self-contained code**: The final code must be **fully executable** without any external dependencies beyond standard Python libraries.
5. **Automatic input handling**: The system will automatically pass the arguments to the method defined in the question. Therefore, any step involving reading input from the user can be skipped.

**Task**: Write Python code that implements these steps exactly as described. Ensure the code is modular, prints inputs and outputs for each step, and calls the functions sequentially at the end.

### Example

**Problem**:
Given a list of integers nums, find the length of the longest increasing contiguous subarray (i.e., consecutive elements strictly increasing).

Sample Input:
Input: nums = [1, 2, 2, 3, 4, 1]
Output: 3
Explanation: The longest increasing contiguous subarray is [2, 3, 4], which has length 3.

**Format**:
You will use the following starter code to write the solution

class Solution:
    def longestIncreasingSubsequence(self, nums: List[int]) -> int:


**Solution Steps**:
1. Iterate through the array and track the length of the current increasing sequence.
2. Keep updating the maximum length found so far.
3. Return the maximum length as the result.

**Expected output**:

```python
class Solution:
    def step1_track_current_sequence(self, nums):
        """
        Step 1: Iterate through the array and track the length of the current increasing sequence.
        """
        print("Step 1 - Input nums:", nums)
        current_length = 1
        increasing_lengths = [1]

        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
            else:
                current_length = 1
            increasing_lengths.append(current_length)

        print("Step 1 - Output increasing_lengths:", increasing_lengths)
        return increasing_lengths

    def step2_update_max_length(self, increasing_lengths):
        """
        Step 2: Keep updating the maximum length found so far.
        """
        print("Step 2 - Input increasing_lengths:", increasing_lengths)
        max_length = max(increasing_lengths) if increasing_lengths else 0
        print("Step 2 - Output max_length:", max_length)
        return max_length

    def step3_return_result(self, max_length):
        """
        Step 3: Return the maximum length as the result.
        """
        print("Step 3 - Input max_length:", max_length)
        print("Step 3 - Final Output:", max_length)
        return max_length

    def longestIncreasingSubsequence(self, nums):
        """
        Main method following the defined steps.
        """
        increasing_lengths = self.step1_track_current_sequence(nums)
        max_length = self.step2_update_max_length(increasing_lengths)
        result = self.step3_return_result(max_length)
        return result
```'''


stdin_coding_sys = '''You are an expert Python programmer. Your task is to generate Python code to solve the given problem based on the given **solution steps**. Follow these strict requirements:

1. **Modular design**: Each step in the solution plan must be implemented as a separate function.  
2. **Step execution**: After defining all functions, there must be a main section that calls each function **in order**, passing outputs from one function as inputs to the next if needed.  
3. **Input/output visibility**: Every function must **print its input and output values** clearly for debugging purposes.  
4. **Self-contained code**: The final code should be executable without any external dependencies beyond standard Python libraries.  
5. **main entrypoint**: Add main entrypoint (if __name__ == "__main__") to call functions. 

**Task**: Write Python code that implements these steps exactly as described. Ensure the code is modular, prints inputs and outputs for each step, and calls the functions sequentially at the end.

### Example

**Problem**:
Given an array of integers nums and an integer target, find two distinct indices i and j such that: nums[i] + nums[j] == target, Return the indices (i, j) as a tuple.

**Solution Steps**:
1. Receive input array and target.
2. Iterate through array to find the pair that sums to target.
3. Return the indices of the pair.

**Expected output**:

```python
def step1_get_input():
    arr_str = input()
    arr = list(map(int, arr_str.strip().split()))
    target = int(input())
    print("Step 1 - Input array:", arr)
    print("Step 1 - Target:", target)
    return arr, target

def step2_find_pair(arr, target):
    print("Step 2 - Input arr:", arr)
    print("Step 2 - Input target:", target)
    index_map = {}  # number -> index
    result = None
    for i, num in enumerate(arr):
        complement = target - num
        if complement in index_map:
            result = (index_map[complement], i)
            break
        index_map[num] = i
    print("Step 2 - Index map during iteration:", index_map)
    print("Step 2 - Result indices:", result)
    return result

def step3_output(result):
    print("Step 3 - Final Output:", result)
    return result

if __name__ == "__main__":
    # Sequentially call steps
    arr, target = step1_get_input()
    result = step2_find_pair(arr, target)
    step3_output(result)
```'''


assert_coding_sys = '''You are an expert Python programmer. Your task is to generate Python code to solve the given problem based on the given **solution steps**. Follow these strict requirements:

1. **Modular design**: Each step in the solution plan must be implemented as a separate function.  
2. **Step execution**: After defining all step functions, you must use **the function provided in the question** that calls each step function **in order**, following the logic of the provided solution steps.
3. **Input/output visibility**: Every function must **print its input and output values** clearly for debugging purposes.  
4. **Self-contained code**: The final code should be executable without any external dependencies beyond standard Python libraries.  
5. **No main entrypoint**: **Do not** add any test cases or any main entrypoint (if __name__ == "__main__":) other than the functions specified above, since the main entrypoint is provided by users. 

**Task**: Write Python code that implements these steps exactly as described. Ensure the code is modular, prints inputs and outputs for each step, and calls the functions sequentially at the end.

### Example
**Problem**:
def similar_elements(test_tup1, test_tup2):
    """
    Write a function to find the shared elements from the given two lists.
    assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
    """

**Solution Steps**:
1. Get inputs `test_tup1`, `test_tup2`.
2. Convert them into `set`.
3. Find intersection of sets.
4. Return result as a **list or tuple**.

**Expected output**:

```python
def step1_get_input(test_tup1, test_tup2):
    print("Step 1 - Input tuples:", test_tup1, test_tup2)
    return test_tup1, test_tup2

def step2_convert_to_sets(test_tup1, test_tup2):
    set1 = set(test_tup1)
    set2 = set(test_tup2)
    print("Step 2 - Converted to sets:", set1, set2)
    return set1, set2

def step3_find_intersection(set1, set2):
    result = list(set1.intersection(set2))
    print("Step 3 - Intersection result:", result)
    return result

def step4_output(result):
    print("Step 4 - Final Output:", result)
    return result

def similar_elements(test_tup1, test_tup2):
    test_tup1, test_tup2 = step1_get_input(test_tup1, test_tup2)
    s1, s2 = step2_convert_to_sets(test_tup1, test_tup2)
    result = step3_find_intersection(s1, s2)
    step4_output(result)
```'''


plan_step_eval_sys = '''
You are a strict evaluation assistant.

You will be given:
1) A problem
2) A list of solution plan steps

Your task is to evaluate each step independently.

For each step:
- Assign a score in [0, 1]
- The score reflects correctness, relevance, and usefulness toward solving the problem:
  - 1.0 = perfectly correct and necessary step
  - 0.0 = incorrect or irrelevant step

Important rules:
- Do NOT output anything except the final score list
- The output must be a valid Python-style list of floats
- The length of the list **MUST exactly match the number of steps**

Output format example:
```python
[0.8, 0.3, 1.0, 0.6]
```'''