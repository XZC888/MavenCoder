from .model import ModelBase, Message
from .prompt import prompt_words
from utils import *
from typing import Tuple, Dict, Any, Optional
from constant import simple_datasets, import_helper


class PyGenerator:
    def __init__(self, logger):
        self.logger = logger
        self.prompts = prompt_words()
        self.reflection = ""

    def _generate_response(
        self,
        model: ModelBase,
        user_content: str,
        system_content: str = "You are a helpful assistant",
        log_prefix: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gen_tokens: bool = False,
    ):
        self.logger.info(f"\nUser:\n{user_content}\n")

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

        tokens = None
        if gen_tokens:
            try:
                tokens, response = model.generate_tokens(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                self.logger.warning(
                    f"generate_tokens failed, fallback to generate_chat. Error: {e}"
                )
                response = model.generate_chat(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
        else:
            response = model.generate_chat(messages=messages, temperature=temperature, top_p=top_p)
        
        prefix = log_prefix if log_prefix else "Assistant"
        self.logger.info(f"\n{prefix}:\n{response}\n[temp={temperature}, top_p={top_p}]\n")
        
        return (response, messages, tokens) if gen_tokens else (response, messages)
    

    def estimate_problem_difficulty(self, item: Dict[str, Any], model: ModelBase, strategy: str):
        user_content = self.prompts.estimate_difficulty_prompt(item['problem'], strategy)

        response, messages, tokens = self._generate_response(
            model=model,
            system_content=self.prompts.ESTIMATION_SYSTEM,
            user_content=user_content,
            gen_tokens=True
        )

        return response, messages, tokens
    

    def self_reflection_thinking(self, item: Dict[str, Any], model: ModelBase) -> Tuple[str, list]:
        user_content = self.prompts.self_reflection_prompt(item['problem'])
        reflection_response, messages = self._generate_response(
            model=model,
            system_content=self.prompts.REFLECTION_SYSTEM,
            user_content=user_content,
        )
        self.reflection = reflection_response

        return reflection_response, messages


    def generate_solution_approach(self, item: Dict[str, Any], model: ModelBase, diff_class: str) -> Tuple[str, list]:
        user_content = self.prompts.planning_prompt(item['problem'], self.reflection, diff_class)
        response, messages, tokens = self._generate_response(
            model=model,
            system_content=self.prompts.PLANNING_SYSTEM,
            user_content=user_content,
            temperature = 0.8,
            top_p=0.95,
            gen_tokens=True
        )
        
        return response, messages, tokens
    

    def implement_code(
        self, 
        item: Dict[str, Any], 
        model: ModelBase, 
        approach: str, 
        dataset_type: str,
    ) -> Tuple[str, list]:
        
        user_content = self.prompts.code_implementation_prompt(
            item['problem'], item['starter_code'], dataset_type, approach
        )
        response, messages = self._generate_response(
            model=model,
            system_content=self.prompts.IMPLEMENTATION_SYSTEM,
            user_content=user_content,
            log_prefix=f"Initial Code",
        )
        
        solution_program = extract_code(response)

        if dataset_type in simple_datasets:
            solution_program = import_helper + "\n" + remove_main(solution_program)
        
        return solution_program, messages


    def generate_repair_instruction(self, item: Dict[str, Any], model: ModelBase, 
                                   execution_results: str, approach: str) -> Tuple[str, list]:
        
        problem = item['problem'] + (f"\n\n## Format:\nYou will use the following starter code to write the solution\n\n{item['starter_code']}" \
                            if item['starter_code'] else "")

        user_content = self.prompts.repair_instruction_prompt(
            problem, execution_results, approach
        )
        response, messages = self._generate_response(
            model=model,
            system_content=self.prompts.REPAIR_EXPLANATION_SYSTEM,
            user_content=user_content,
            log_prefix="Fix Explanation"
        )
        
        return response, messages


    def repair_buggy_code(self, iteration: int, item: Dict[str, Any], model: ModelBase, 
                            error_code: str, explanation: str, approach: str, dataset_type: str) -> Tuple[str, list]:
        
        problem = item['problem'] + (f"\n\n## Format:\nYou will use the following starter code to write the solution\n\n{item['starter_code']}" if item['starter_code'] else "")

        user_content = self.prompts.repair_code_prompt(
            error_code, problem, explanation, approach,
        )
        response, messages = self._generate_response(
            model=model,
            system_content=self.prompts.REPAIR_CODE_SYSTEM,
            user_content=user_content,
            log_prefix=f"New Repair Code{iteration}"
        )
        
        repaired_code = extract_code(response)

        if dataset_type in simple_datasets:
            repaired_code = import_helper + "\n" + remove_main(repaired_code)

        return repaired_code, messages
    

    def plan_refinement(
        self, 
        item: Dict[str, Any], 
        model: ModelBase, 
        review: str,
        approach: str,
    ) -> Tuple[str, list]:
        
        user_content = self.prompts.rebuild_approach(item['problem'], review, approach)
        response, messages, tokens = self._generate_response(
            model=model,
            system_content=self.prompts.PLANNING_SYSTEM,
            user_content=user_content,
            gen_tokens=True
        )

        return response, messages, tokens
