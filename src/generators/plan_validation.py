import ast
from typing import *
import yaml
from scipy.special import softmax
from scipy.stats import beta
import numpy as np
from typing import *
import json
import subprocess
import tempfile
import os
import sys

from constant import fact_sys, call_based_coding_sys, stdin_coding_sys, assert_coding_sys, plan_step_eval_sys
from utils import extract_steps, extract_code


class StepAnalyzer:
    """
    Step Reliability
    """
    @staticmethod
    def compute_token_confidence(tokens) -> List[float]:
        return [
            round(-np.mean([lp.logprob for lp in t.top_logprobs]), 3)
            for t in tokens
        ]


    @staticmethod
    def is_template_token(token_text: str) -> bool:
        if not token_text.strip():
            return True
        return any(p in token_text for p in ["\n", "```", "**", "#", "-"])
    

    @classmethod
    def step_confidence(cls, plan_steps, plan_tokens):
        generated_text = "".join([t.token for t in plan_tokens])
        step_confidences = []
        pointer = 0

        for step in plan_steps:
            start = generated_text.find(step, pointer)
            end = start + len(step)
            pointer = end

            cur_tokens = []
            char_pos = 0
            for tok in plan_tokens:
                if cls.is_template_token(tok.token):
                    continue

                tok_start = char_pos
                tok_end = char_pos + len(tok.token)

                if tok_start >= start and tok_end <= end:
                    cur_tokens.append(tok)

                char_pos = tok_end

            if cur_tokens:
                avg_conf = np.mean(cls.compute_token_confidence(cur_tokens))
                step_confidences.append(round(avg_conf, 3))

        return step_confidences



class CodeExecutor:
    """
    Factual Correctness
    """
    @staticmethod
    def run_code_stdin(code: str, tests: List[str], timeout: int = 6) -> List[str]:
        results = []
        
        for test_input in tests:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    input=test_input,
                    text=True,
                    capture_output=True,
                    timeout=timeout
                )
                output = result.stdout
                results.append(output)
            except subprocess.TimeoutExpired:
                results.append("Time Limited Error: Code execution timed out\n")
            except Exception as e:
                results.append(f"Execution Error: {str(e)}\n")
            finally:
                os.unlink(temp_file)
        
        return results


    @staticmethod
    def run_code_called(code: str, tests: List[str], func_name: str, timeout: int = 6):
        outputs = []

        for test_input in tests:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmp:
                tmp.write(code)
                tmp.write(f"""

import json
import sys

sol = Solution()
lines = sys.stdin.read().splitlines()
args = [json.loads(x) for x in lines if x.strip()]
result = getattr(sol, "{func_name}")(*args)
print(result)
""")
                tmp_path = tmp.name

            try:
                proc = subprocess.Popen(
                    [sys.executable, tmp_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                stdout, stderr = proc.communicate(input=test_input, timeout=timeout)

                if stderr:
                    outputs.append(f"Execution Error: {stderr}")
                else:
                    outputs.append(stdout.strip())

            except subprocess.TimeoutExpired:
                proc.kill()
                outputs.append("Execution Error: Timeout")
            finally:
                os.remove(tmp_path)

        return outputs


    @staticmethod
    def run_code_assert(code: str, tests: List[str], timeout: int = 6) -> List[str]:
        results = []
        
        for test_input in tests:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.write(f"""

if __name__ == "__main__":
    {test_input}
""")
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    input=test_input,
                    text=True,
                    capture_output=True,
                    timeout=timeout
                )
                output = result.stdout
                results.append(output)
            except subprocess.TimeoutExpired:
                results.append("Time Limited Error: Code execution timed out\n")
            except Exception as e:
                results.append(f"Execution Error: {str(e)}\n")
            finally:
                os.unlink(temp_file)
        
        return results


class Plan_Valifier:
    def __init__(self, gen, model, item, competitive_type=True):
        self.gen = gen
        self.model = model
        self.item = item
        self.competitive_type = competitive_type
        self.assert_coding_sys = assert_coding_sys
        self.stdin_coding_sys = stdin_coding_sys
        self.call_based_coding_sys = call_based_coding_sys
        self.fact_sys = fact_sys


    def get_solution_plan(self, solution_plan):
        self.plan_steps = extract_steps(solution_plan)


    def caculate_confidence(self, plan_tokens):
        n_steps = len(self.plan_steps)

        if plan_tokens:
            C = StepAnalyzer.step_confidence(self.plan_steps, plan_tokens)
            return softmax(C)

        # LLM fallback
        print("plan tokens not support, fallback to LLM evaluation for step reliability.")
        
        steps = "\n".join(
            f"Step {i+1}: {step}"
            for i, step in enumerate(self.plan_steps)
        )

        response, _ = self.gen._generate_response(
            model=self.model,
            user_content=(
                f"Problem:\n{self.item['problem']}\n\n"
                f"Plan Steps:\n{steps}\n\n"
                f"Return a Python list of {n_steps} scores in [0, 1]."
            ),
            system_content=plan_step_eval_sys,
        )

        try:
            raw = extract_code(response)
            C = ast.literal_eval(raw)

            if not isinstance(C, list) or len(C) != n_steps:
                raise ValueError("Invalid length or type")
            return np.array(C)

        except Exception as e:
            raise ValueError(f"Failed to parse confidence scores: {response}") from e
    

    def gen_modular_code(self):
        steps = "\n".join(self.plan_steps)

        problem = self.item['problem']

        if self.item['starter_code']: # call-based
            problem += f"\n\n## Format:\nYou will use the following starter code to write the solution\n\n{self.item['starter_code']}"
            sys = self.call_based_coding_sys
        elif self.competitive_type:
            sys = self.stdin_coding_sys
        else:
            sys = self.assert_coding_sys

        response, messages = self.gen._generate_response(
            model = self.model, 
            user_content = f"""Generate a modual code for the {len(self.plan_steps)} steps below.\n\n### Problem:\n{problem}\n\n### Solution Steps:\n{steps}""", 
            system_content = sys,
            log_prefix=f"Modual Code",
        )

        return extract_code(response), messages
    

    def caculate_fact_values(self, code):
        if self.competitive_type:
            in_outs = json.loads(self.item["public_test_cases"]["input_output"])
            samples = in_outs["inputs"]

            if self.item["starter_code"]: # call based code 
                outputs = CodeExecutor.run_code_called(code, samples, in_outs.get("fn_name"))
            else:
                outputs = CodeExecutor.run_code_stdin(code, samples)
        else:
            samples = self.item["public_test_cases"]
            outputs = CodeExecutor.run_code_assert(code, samples)
        
        steps_text = "\n".join(self.plan_steps)
        problem = self.item["problem"]
        n_steps = len(self.plan_steps)

        F_values = []
        responses = []
        messages_list = []

        for sample, output in zip(samples, outputs):
            prompt = (
                f"Please evaluate the correctness of each **{n_steps} steps** below.\n\n"
                f"### Problem:\n{problem}\n\n"
                f"### Solution Steps:\n{steps_text}\n\n"
                f"### Input:\n{sample}\n\n"
                f"### Execution Result:\n{output[:900]}"
            )

            response, messages = self.gen._generate_response(
                self.model, 
                prompt,
                self.fact_sys, 
                "Facts Eval"
            )
            responses.append(response)
            messages_list.append(messages)

            fact_yaml = yaml.safe_load(extract_code(response, "yaml"))

            scores = [r["score"] for r in fact_yaml["steps"]]
            F_values.append(scores)

        F = np.mean(np.array(F_values), axis=0)
        return F, responses, messages_list
    

    def caculate_weights_values(self, alpha=3, beta_p=5):
        x = np.linspace(0, 1, len(self.plan_steps))
        w_raw = beta.pdf(x, alpha, beta_p)
        return np.round(1 - w_raw / w_raw.sum(), 3)

