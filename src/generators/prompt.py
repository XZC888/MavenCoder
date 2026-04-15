class prompt_words:
    def __init__(self):
        self.simple_datasets = ["humanevalplus", "mbppplus"]

        self.ESTIMATION_SYSTEM = "You are an expert in evaluating problem difficulty for competitive programming problems."

        self.REFLECTION_SYSTEM = "You are an expert problem analyst."

        self.PLANNING_SYSTEM = f"You are an expert programming problem solver and algorithm designer."

        self.IMPLEMENTATION_SYSTEM = "You're an expert in solving programming problems. Generate correct solution based on provided approach"

        self.REPAIR_EXPLANATION_SYSTEM = "You're an expert in debugging and provide useful fix suggestion"

        self.REPAIR_CODE_SYSTEM = "You're an expert in debugging and fixing buggy codes"

        self.VALIDATE_SYSTEM = "You are a rigorous programming validator. Analyze the given problem and correctness of solution approach through critical verification." 


    def estimate_difficulty_prompt(self, problem, strategy):               
        p = f'''You are an AI assistant that evaluates the difficulty of programming problems. 

### Evaluation Dimensions
Evaluate the difficulty based on the following aspects:
1. **Understanding complexity** - How hard is it to correctly interpret the problem?
2. **Reasoning depth** - How many logical steps are required to solve it?
3. **Algorithmic sophistication** - What is the expected algorithmic or conceptual difficulty?
4. **Mathematical difficulty** - How much mathematical reasoning, formulas, or abstract modeling is required.
5. **implementation_difficulty** - How hard it is to implement, test, and debug the solution.

### Problem:
{problem}'''
    
        if strategy == "prompt":
            p += '''\n\n\n**output only the difficulty level as one of the following categories without any explanations:**
- easy
- medium
- hard'''
        return p


    def self_reflection_prompt(self, problem):
        return f'''Given a programming problem, analyze the problem through these steps to provide a comprehensive analysis.
        
1. **Problem Comprehension**
    - Understand the intend of this problem
    - Identify core requirements.
            
2. **Example Explanation**
For each provided example, explain how each sample input will yield the sample output through logical step-by-step inference.

3. **Edge Case Synthesis**:
    - Consider data constraints and inform how to properly handle edge cases
    
[`Problem`]:
{problem}

Now, provide your comprehensive problem analysis'''
    

    def planning_prompt(self, problem, reflection, diff_class) -> str:        
        reflection = f"**Your reflection for this problem:**\n{reflection}\n" if reflection else ""

        if "easy" in diff_class:
             return f'''You are given a programmming problem:
### Problem:
{problem.strip()}

Please briefly give a concrete explanation for the problem, and provide logical steps without codes to solve the problem. Avoid overthinking. Sequence operations logically to solve the problem, start with ### Solution Steps.

That's think step by step for the given problem.'''

        return f"""You are given a programmming problem:
### Problem:
{problem.strip()}

{reflection}
Provide a solution approach with natural language without codes.

You should:
1. Understand the intend of problem and analyze key requirements from problem statement
2. Explain how public sample inputs yield the expected outputs.
3. Think of proper algorithm tags, map each requirement to relevant algorithm tags and identify necessary operations implied by the tags
4. Sequence operations logically to solve the problem.

**Response Format:**

### Problem Understanding:
..
### Inference Examples:
..
### Algorithm Tags:
..
### Solution Steps:
1. (the first step here)...
2. (the second step here)
...

That's think step by step for the given problem."""
# Now, generate a solution approach for the given problem.


    def code_implementation_prompt(self, question_content, starter_code, dataset_type, approach="", language="python"):
        FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

        FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

        prompt = f"### Question:\n{question_content.strip()}\n\n"
        if approach:
            prompt += f"### Approach:\n{approach}\n\n"

        if dataset_type in self.simple_datasets:
            return prompt

        if starter_code:
            prompt += (
                f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            )
            prompt += f"```{language}\n{starter_code}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            prompt += f"```{language}\n# YOUR CODE HERE\n```\n\n"
        prompt += f"### Answer: (use the provided format with backticks)\n\n"

        return prompt
    

    def repair_instruction_prompt(self, problem, execution_results, approach):    
        return f"""**Task**: You are given a programming problem and a possible implement approach. Please analyze programming errors and provide natural-language repair guidance for the last buggy code.

`[Problem]`:
{problem}

`[Approach]`:
{approach}

`[Submission History]`:
{execution_results}

First, ensure the code correctly accepts input and produces the proper output format, for instance, if the function call is missing, the actual output of given code will be empty, and you should suggest to call it explicitly.

Second, you should clearly identify the **root cause** of failure in the implementations, specify **exact location(s)** of critical flaws and explain **why** these cause failure with concrete evidence from: problem constraints, historical commit context and failure scenarios. 
Then, provide **sequential repair instructions** that specify content to fix and ensure functional correctness.

**Output Format**:
**Analysis**: [Concise error explanation]
**Critical Fault Points**:
    - [Location]: [Description of flaw]
**Repair Steps**:
1. [Action 1 targeting specific code segment]
    Rationale: [How this resolves root cause]
2. [Action 2 addressing secondary issues]
    Rationale: [Prevention of related failures]
..."""
    

    def repair_code_prompt(self, error_code, problem, explanation, approach):        
        return f"""You are given a problem:
{problem}

A Buggy Implementation for the problem:
{error_code}

The Approach that buggy codes implements:

{approach}

Concrete Fix Instructions:
{explanation}

Your Task is to generate a corrected, runnable program that EXACTLY implements the provided fix instructions, ensure the solution addresses all issues mentioned in the fix instructions and pass all hidden test cases (**Your code should not directly test on the sample inputs**). Output ONLY the corrected Python code without any explanations, comments, or additional text."""


    def rebuild_approach(self, problem, review, approach):
        return f'''Given a programming problem, you have generated a solution approach for the problem. However, you reviewd this approach and found some mistakes. Please generated a new correct natural-language approach without codes.
        
You should avoid to repeat the same logic errors in the original approach.

[`Problem`]:
{problem}

[`Original Approach`]:
{approach}

[`Your Review Explanation`]:
{review}


**Response Format:**

### Problem Understanding:
..
### Inference Examples:
..
### Algorithm Tags:
..
### Solution Steps:
1. (the first step here)...
2. (the second step here)
...'''
# Now, based on the above requirements, generate a new correct approach for the problem.
