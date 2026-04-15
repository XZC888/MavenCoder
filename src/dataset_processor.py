from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import os
from utils import read_jsonl
from constant import competitive_datasets, simple_datasets

class BaseDatasetProcessor(ABC):
    @abstractmethod
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        convert raw data item into the unified format。
            - task_id: str
            - problem: str
            - public_test_cases: list or dict
            - starter_code: str
            - metadata: dict
        """
        raise NotImplementedError()


class Competitive_Processor(BaseDatasetProcessor):
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = json.loads(item["metadata"]) if "metadata" in item else {}
        public_test_cases = json.loads(item["public_test_cases"])
        sample = {
            "input_output": json.dumps({
                "inputs": [t["input"] for t in public_test_cases],
                "outputs": [t["output"] for t in public_test_cases],
                "fn_name": metadata.get("func_name", None),
            }),
        }
        return {
            "task_id": item["question_id"],
            "problem": item["question_content"],
            "public_test_cases": sample,
            "starter_code": item.get("starter_code", ""),
            "metadata": metadata,
        }

class Simple_Processor(BaseDatasetProcessor):
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": item["task_id"],
            "problem": item["prompt"],
            "public_test_cases": item["public_test_cases"],
            "starter_code": "",
            "metadata": {
                "func_name": item["entry_point"]
            }
        }



def processed_dataset(dataset: str) -> List[dict]:
    dataset_path = os.path.join("./data", f"{dataset}.jsonl")

    if dataset in competitive_datasets:
        processor = Competitive_Processor()
    elif dataset in simple_datasets:
        processor = Simple_Processor()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    datasets = []
    for item in read_jsonl(dataset_path):
        datasets.append(processor.process(item))

    return datasets


def convert_format(file_path, dataset, test_dir):
    """
    convert output file to specific evaluation format, see official repository `Evalplus` and `LiveCodeBench` for details.
    """
    os.makedirs(test_dir, exist_ok=True)
    
    if dataset in competitive_datasets:
        output_file = os.path.join(test_dir, os.path.basename(file_path).replace(".jsonl", ".json"))
        json_list = []
        
        for line in read_jsonl(file_path):
            json_object = {
                "question_id": line["task_id"],
                "code_list": [line["solution"]],
            }
            json_list.append(json_object)
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(json_list, f_out, ensure_ascii=False, indent=2)

    elif dataset in simple_datasets:
        output_file = os.path.join(test_dir, os.path.basename(file_path))
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in read_jsonl(file_path):
                f_out.write(json.dumps({"task_id": line["task_id"], "solution": line["solution"]}) + "\n")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")