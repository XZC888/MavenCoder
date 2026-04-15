from collections import deque
import re
import re
import numpy as np
from constant import RESOLVE_SYSTEM, PLAN_DESIGN_SYSTEM, CONQUER_SYSTEM


def build_resolve_prompt(problem: str, node, reference_text: str) -> str:
    if reference_text:
        return f"""### Problem to Solve:
{problem}

### Current Problem:
{node.cur_problem}

{reference_text}
Now, using the information above, provide the final reasoning and integrated solution."""
    
    return f"""### Problem to Solve:
{problem}

### Current Problem:
{node.cur_problem}

Provide a detailed reasoning-based solution for the current problem."""


def build_reference_text(node, all_sub_problems_vectors) -> str:
    if not node.sub_problems:
        return ""

    ref = "\n\n### Reference Sub-Problems and Solutions:\n"

    for sub_p in node.sub_problems:
        for old_text, _, old_sol in all_sub_problems_vectors:
            if old_text == sub_p and old_sol.strip():
                ref += f"- Sub-problem: {old_text}\n  Solution: {old_sol}\n\n"
    return ref


class Node:
    def __init__(self, cur_problem, solution="", depth=0):
        self.cur_problem = cur_problem
        self.solution = solution
        self.children = []
        self.depth = depth
        self.sub_problems = []

class Plan_Tree:
    def __init__(self, model, similarity_threshold=0.7, max_depth=5):
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.all_sub_problems_vectors = []
        self.model = model

        self.RESOLVE_SYSTEM = RESOLVE_SYSTEM
        self.PLAN_DESIGN_SYSTEM = PLAN_DESIGN_SYSTEM
        self.CONQUER_SYSTEM = CONQUER_SYSTEM


    def _is_duplicate(self, new_vec):
        for old_text, old_vec, _ in self.all_sub_problems_vectors:
            similarity = np.dot(old_vec, new_vec) / (np.linalg.norm(old_vec) * np.linalg.norm(new_vec))
            if similarity > self.similarity_threshold:
                return True, similarity, old_text
        return False, 0, ""


    # Problem Decomposition
    def _extract_sub_problems(self, response):
        pattern = r"### Sub-Problems:\s*([\s\S]*?)(?=### |$)"
        match = re.search(pattern, response)
        if match:
            sub_block = match.group(1).strip()
            sub_list = [
                s.strip() for s in sub_block.split("\n")
                if s.strip() and s.strip()[0] in "0123456789"
            ]
            return sub_list
        return []


    # Construct Plan Tree
    def generate_solution_plan(self, model, problem, gen):
        root = Node(problem)

        responses, messages = [], []
        response, message1 = gen._generate_response(model, f"### Current Problem:\n{problem}", self.PLAN_DESIGN_SYSTEM, "Root Problem:", 0.8, 0.95)
        responses.append(response)
        messages.append(message1)

        sub_list = self._extract_sub_problems(response)
        if sub_list:
            root.sub_problems = sub_list
            vectors = self.model.get_embedding(sub_list)
            self.all_sub_problems_vectors.extend([(s, v, "") for s, v in zip(sub_list, vectors)])
            root.children = [Node(p, depth=1) for p in sub_list]
        else:
            root.solution = response

        queue = deque(root.children)
        while queue:
            node = queue.popleft()
            if node.depth >= self.max_depth:
                continue

            response, message2 = gen._generate_response(
                model,
                f"### Problem to Solve:\n{problem}\n\n### current problem:\n{node.cur_problem}", 
                self.PLAN_DESIGN_SYSTEM
            )
            responses.append(response)
            messages.append(message2)

            sub_list = self._extract_sub_problems(response)
            node.sub_problems = sub_list

            if sub_list:
                for sub_p in sub_list:
                    sub_vec = self.model.get_embedding(sub_p)
                    is_dup, sim, sim_p = self._is_duplicate(sub_vec)
                    if not is_dup:
                        self.all_sub_problems_vectors.append((sub_p, sub_vec, ""))
                        node.children.append(Node(sub_p, depth=node.depth + 1))
                queue.extend(node.children)
            else:
                node.solution = response
                for i, (text, vec, sol) in enumerate(self.all_sub_problems_vectors):
                    if text == node.cur_problem:
                        self.all_sub_problems_vectors[i] = (text, vec, response)
                        break

        return root, responses, messages


    def resolve_unfinished_leaves(self, root, model, problem, gen):
        queue = deque([root])
        messages = []

        while queue:
            node = queue.popleft()

            if node.children:
                queue.extend(node.children)
                continue

            if not node.solution.strip():
                reference_text = build_reference_text(
                    node, self.all_sub_problems_vectors
                )

                prompt = build_resolve_prompt(
                    problem, node, reference_text
                )

                response, message = gen._generate_response(
                    model,
                    prompt,
                    self.RESOLVE_SYSTEM,
                    "Resolved leaf:"
                )

                messages.append(message)

                node.solution = (
                    f"### Current Problem:\n{node.cur_problem}\n\n{response}"
                )

        return messages


    def serialize_plan_tree(self, node: Node, indent: int = 0, index: str = "") -> str:
        prefix = "  " * indent
        result = ""

        if indent == 0:
            result += f"To solve this problem, we can further break it down into more sub-problems:\n"
        else:
            result += f"{prefix}- sub-problem {index}:\n"
            if not node.solution:
                lines = node.cur_problem.strip().split("\n")
                for line in lines:
                    result += f"{prefix}  {line}\n"
                result += "\n"

        if not node.children:
            if node.solution:
                lines = node.solution.strip().split("\n")
                for line in lines:
                    result += f"{prefix}  {line}\n"
                result += "\n"
            return result

        for i, child in enumerate(node.children, 1):
            new_index = f"{index}-{i}" if index else str(i)
            result += self.serialize_plan_tree(child, indent + 1, new_index)

        return result


    def generate_final_solution_plan(self, model, flatten_plan, problem, gen):
        return gen._generate_response(
            model=model, 
            user_content = f"### Problem to Solve:\n{problem}\n\n\n### Plan Tree:\n\n{flatten_plan}", 
            system_content = self.CONQUER_SYSTEM, 
            log_prefix = "Final Plan:", 
            gen_tokens=True
        )
