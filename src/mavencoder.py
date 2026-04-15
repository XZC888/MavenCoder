import os
import numpy as np
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from tqdm import tqdm

from generators import PyGenerator, model_factory, Plan_Tree, Plan_Valifier
from executors import run_tests
from log_helper import setup_logger
from utils import *
from conf import *
from dataset_processor import processed_dataset
from constant import simple_datasets



@dataclass
class Config:
    dataset_type: str
    model_name: str
    strategy: str
    theta_1: float
    theta_2: float
    r_global: int
    r_debug: int
    r_valid: int
    output_path: str
    log_dir: str
    key: str
    url: str
    embedding_key: str
    embedding_url: str
    embedding_model: str
    verbose: bool


@dataclass
class Context:
    item: dict
    model: Any
    gen: Any
    config: Config
    logger: Any


class TokenTracker:
    def __init__(self):
        self.stats = {
            "classification_tokens": 0,
            "reflection_tokens": 0,
            "solution_approach_tokens": 0,
            "code_generation_tokens": 0,
            "debug_tokens": 0,
        }

    def add(self, key: str, val: int):
        self.stats[key] += val

    def get(self):
        return self.stats


def adaptive_difficulty_assessment(ctx: Context) -> Tuple[str, int]:
    cfg = ctx.config

    pred, msg, tokens = ctx.gen.estimate_problem_difficulty(
        ctx.item, ctx.model, cfg.strategy
    )

    if cfg.strategy != "prompt":
        token_conf = compute_token_confidence(tokens, cfg.strategy)
        infer_conf = compute_inference_confidence(tokens, token_conf)
        pred = adaptive_predict(infer_conf, cfg.theta_1, cfg.theta_2)

    assert pred in ["easy", "medium", "hard"]
    return pred, count_tokens(pred, msg)


def self_reflection(ctx: Context):
    reflection, msg = ctx.gen.self_reflection_thinking(ctx.item, ctx.model)
    return reflection, count_tokens(reflection, msg)


def plan_tree_planning(ctx: Context):
    """Tree-based task decomposition and exploration"""

    pt = Plan_Tree(ctx.model)

    root, responses, msgs = pt.generate_solution_plan(
        ctx.model, ctx.item["problem"], ctx.gen
    )

    token_num = sum(count_tokens(r, m) for r, m in zip(responses, msgs))

    resolve_msgs = pt.resolve_unfinished_leaves(
        root, ctx.model, ctx.item["problem"], ctx.gen
    )
    token_num += sum(count_tokens(m) for m in resolve_msgs)

    flat = pt.serialize_plan_tree(root)

    sol, msg, plan_tokens = pt.generate_final_solution_plan(
        ctx.model, flat, ctx.item["problem"], ctx.gen
    )

    token_num += count_tokens(flat, msg)

    return sol, token_num, plan_tokens


def plan_verification(ctx: Context, solution_approach: str, plan_tokens):
    token_num = 0

    competitive_type = ctx.config.dataset_type not in simple_datasets

    pv = Plan_Valifier(ctx.gen, ctx.model, ctx.item, competitive_type)
    pv.get_solution_plan(solution_approach)

    for _ in range(ctx.config.r_valid):
        C = pv.caculate_confidence(plan_tokens)
        W = pv.caculate_weights_values()

        modular_code, code_msg = pv.gen_modular_code()
        token_num += count_tokens(modular_code, code_msg)

        F, _, fact_msgs = pv.caculate_fact_values(modular_code)
        token_num += sum(count_tokens(m) for m in fact_msgs)

        threshold = 0.5 * (1 / (len(C) + 5)) * np.min(W)

        if not (len(C) == len(F) == len(W)):
            raise ValueError(
                f"Length mismatch: C={len(C)}, F={len(F)}, W={len(W)}"
            )

        pos = None
        for i, (c, f, w) in enumerate(zip(C, F, W), start=1):
            if c * f * w < threshold:
                pos = i
                break

        if pos is None:
            # correct plan
            return solution_approach, token_num

        # refinement
        solution_approach, review_msg, plan_tokens = ctx.gen.plan_refinement(
            ctx.item,
            ctx.model,
            f"the **Step{pos}** may be incorrect, please correct the mistakes in it if you find.",
            solution_approach,
        )
        token_num += count_tokens(solution_approach, review_msg)

    return solution_approach, token_num


def planning_stage(ctx: Context, diff: str):
    if diff == "hard":
        sol, token_num, plan_tokens = plan_tree_planning(ctx)
    else:
        sol, msg, plan_tokens = ctx.gen.generate_solution_approach(
            ctx.item, ctx.model, diff
        )
        token_num = count_tokens(sol, msg)

    sol, token_verification = plan_verification(ctx, sol, plan_tokens)
    token_num += token_verification

    return sol, token_num


def generate_code(ctx: Context, approach: str):
    code, msg = ctx.gen.implement_code(
        ctx.item, ctx.model, approach, ctx.config.dataset_type
    )
    return code, count_tokens(code, msg)


def debug_loop(ctx: Context, code: str, error: str, approach: str, logger):
    token_total = 0
    history_parts = []

    current_code, current_error = code, error

    for i in range(1, ctx.config.r_debug + 1):
        history_parts.append(
            f"code{i}:\n{current_code.strip()}\n\n{current_error.strip()}\n\n"
        )
        history = "".join(history_parts)

        explain, msg = ctx.gen.generate_repair_instruction(
            ctx.item, ctx.model, history, approach
        )
        token_total += count_tokens(explain, msg)

        repaired_code, msg = ctx.gen.repair_buggy_code(
            i, ctx.item, ctx.model,
            current_code, explain, approach, ctx.config.dataset_type
        )
        token_total += count_tokens(repaired_code, msg)

        passed, new_error = run_tests(
            repaired_code, ctx.item, ctx.config.dataset_type
        )

        result = new_error if new_error else "Passed"
        logger.info(f"Repaired code{i} test result:\n{result}")

        if passed:
            return repaired_code, token_total, True, i

        current_code, current_error = repaired_code, new_error

    return current_code, token_total, False, ctx.config.r_debug


def update_item(ctx: Context, code: str, success: bool, debug_iter: int, tracker: TokenTracker):
    ctx.item["solution"] = code
    ctx.item["passing_public_tests"] = success
    ctx.item["debug_iter"] = debug_iter
    ctx.item["tokens"] = tracker.get()


def mavencoder_task(item: dict, config: Config):
    logger = setup_logger(
        os.path.join(config.log_dir, f"{item['task_id'].replace('/', '_')}.log"),
        config.verbose,
    )

    gen = PyGenerator(logger)
    model = model_factory(
        config.model_name,
        key=config.key,
        url=config.url,
        embedding_key=config.embedding_key,
        embedding_url=config.embedding_url,
        embedding_model=config.embedding_model,
    )

    ctx = Context(item=item, model=model, gen=gen, config=config, logger=logger)
    tracker = TokenTracker()

    diff, tokens = adaptive_difficulty_assessment(ctx)
    tracker.add("classification_tokens", tokens)
    ctx.item["diff_class"] = diff
    logger.info(f"Difficulty Assessment: {diff}\n")

    # reflection
    if diff == "medium":
        reflection, tokens = self_reflection(ctx)
        tracker.add("reflection_tokens", tokens)
        ctx.item["reflection"] = reflection

    for r in range(1, config.r_global + 1):
        approach, tokens = planning_stage(ctx, diff)
        tracker.add("solution_approach_tokens", tokens)

        code, tokens = generate_code(ctx, approach)
        tracker.add("code_generation_tokens", tokens)

        passed, error = run_tests(code, ctx.item, config.dataset_type)
        logger.info(f"Initial Code Test Result:\n{error if error else 'Passed'}")

        if passed:
            logger.info("-----------------")
            logger.info(f"{ctx.item['task_id']} pass public tests.")
            update_item(ctx, code, True, 0, tracker)
            write_jsonl(ctx.config.output_path, [ctx.item], append=True)
            return

        code, tokens, success, iters = debug_loop(ctx, code, error, approach, logger)
        tracker.add("debug_tokens", tokens)
        update_item(ctx, code, success, iters, tracker)

        logger.info("-----------------")
        if success:
            logger.info(f"{ctx.item['task_id']} pass public tests.")
            write_jsonl(ctx.config.output_path, [ctx.item], append=True)
            return

        logger.info(f"{ctx.item['task_id']} failed public tests after {r} iterations.")

    write_jsonl(ctx.config.output_path, [ctx.item], append=True)


def run(
    dataset_type: str,
    model_name: str,
    strategy: str,
    theta_1: float,
    theta_2: float,
    r_global: int,
    r_debug: int,
    r_valid: int,
    output_path: str,
    log_dir: str,
    key: str,
    url: str,
    embedding_key: str,
    embedding_url: str,
    embedding_model: str,
    verbose: bool,
    max_workers: int,
):
    """main pipline for MavenCoder"""
    logger = setup_logger(os.path.join(log_dir, "global.log"), verbose, "a")

    ids = set(load_ids(output_path))

    logger.info("Loading dataset...")
    dataset = processed_dataset(dataset_type)
    logger.info(f"Loaded {len(dataset)} examples")

    config = Config(
        dataset_type, model_name, strategy, theta_1, theta_2, r_global, r_debug, r_valid,
        output_path, log_dir, key, url, embedding_key, embedding_url, embedding_model, verbose,
    )

    tasks = [item for item in dataset if item["task_id"] not in ids]
    logger.info(f"Remain {len(tasks)} tasks")

    pbar = tqdm(total=len(tasks), desc="Running", dynamic_ncols=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(mavencoder_task, item, config): item["task_id"]
            for item in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in {task_id}: {e}")
            pbar.update(1)

    pbar.close()

    assert len(dataset) == len(load_ids(output_path)), "Some tasks are missing, run again."
    logger.info(f"Public Tests Accuracy: {count_solved(output_path)}")