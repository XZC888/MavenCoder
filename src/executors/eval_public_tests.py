from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
from .simple_eval import run_assert
from .competitive_eval import run_stdin
from src.constant import competitive_datasets, simple_datasets

def run_tests(code: str, item: dict, dataset_type: str) -> Tuple[bool, str]:
    if dataset_type in simple_datasets:
        test_func = run_assert
    elif dataset_type in competitive_datasets:
        test_func = run_stdin
    else:
        raise RuntimeError(f"Unknown dataset type: {dataset_type}")

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_func, item['public_test_cases'], code)
        _, error_message, passed = future.result()

    return passed, error_message