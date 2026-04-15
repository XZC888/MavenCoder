import os
import json
import jsonlines
import re
from typing import List, Dict
import tiktoken


IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\nimport hashlib\nimport heapq\nimport collections\nfrom collections import *\nfrom itertools import combinations\nfrom math import prod\nfrom itertools import combinations_with_replacement\nfrom  decimal import Decimal, getcontext\nimport numpy as np\n"


def count_tokens(*texts):
    encoder = tiktoken.encoding_for_model("gpt-4")
    count = 0
    for text in texts:
        if isinstance(text, list):
            count += sum(len(encoder.encode(msg.content)) for msg in text)
        else:
            count += len(encoder.encode(text))
    return count



def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def count_solved(output_path) -> float:
    solved = 0
    count = 0
    dataset = open(output_path, "r")
    for l in dataset:
        item = json.loads(l)
        count += 1
        if "passing_public_tests" in item and item["passing_public_tests"]:
            solved += 1
    return float(solved) / count


def load_ids(file_path):
    ids = set()
    if not os.path.exists(file_path):
        return ids
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = json.loads(line)
            ids.add(line["task_id"])
    return ids


def remove_main(code: str) -> str:
    for identifier in ['\n#', '\nif', '\nassert']:
        if identifier in code:
            code = code.split(identifier)[0]
    return code


def extract_code(response, language="python"):
    pattern = rf"```{language.lower()}(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern = r"```(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()



def extract_steps(solution_plan):
    keywords = ["Solution Steps", "Solution Approach", "Step-by-step Approach"]

    start_idx = -1
    keyword = None

    for kw in keywords:
        idx = solution_plan.rfind(kw)
        if idx != -1:
            start_idx = idx
            keyword = kw
            break

    if start_idx == -1:
        raise ValueError("No solution steps found.")

    steps_block = solution_plan[start_idx + len(keyword):]

    lines = steps_block.split("\n")

    steps = []
    current_step = None

    for line in lines:
        stripped = line.strip()

        if stripped and (stripped.split('.', 1)[0].isdigit() or bool(re.match(r"^#+\s*\d+\.", stripped))):
            if current_step:
                steps.append(current_step.strip())
            current_step = stripped
        else:
            if current_step is not None:
                current_step += " " + stripped

    if current_step:
        steps.append(current_step.strip())

    return steps