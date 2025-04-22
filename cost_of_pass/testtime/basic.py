from textwrap import dedent
from cost_of_pass.testtime.base import TestTimeMethod, register_test_time_method
import re
from typing import Callable

@register_test_time_method("VanillaPromptMethod")
class VanillaPromptMethod(TestTimeMethod):
    prompt_template = dedent(\
    """
    Please solve the following question. You can explain your solution before presenting the final answer.

    Format your final answer as:

    <answer>
    ...
    </answer>

    Instructions:
    - For multiple-choice: Give only the letter (e.g., (A)).
    - For numeric: Give only the number (e.g., 42).
    - For free-response: Provide the full final answer text.

    INPUT:
    '''
    {input}
    '''
    """)

    def __init__(self, method_name="VanillaPromptMethod"):
        super().__init__(method_name)

    def extract_answer(self, output: str) -> str:
        matches = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)
        return matches[-1].strip() if matches else ""
    
    def run_for_one(self, client_query_fun: Callable, input: str, comparator: Callable) -> str:
        prompt = self.prompt_template.format(input=input)
        output = client_query_fun(prompt)
        return self.extract_answer(output)



@register_test_time_method("SelfRefinementMethod")
class SelfRefinementMethod(VanillaPromptMethod):
    
    self_refinement_prompt = dedent(\
    """
    Review the response below. First, verify the explanation and final answer. If incorrect, fix them.

    Then, rewrite the explanation to be clearer, more concise, and logically sound.

    Format:

    <answer>
    ...
    </answer>

    Instructions:
    - Multiple-choice: Only the letter (e.g., (A))
    - Numeric: Only the number (e.g., 42)
    - Free-response: Full final answer text

    INPUT:
    '''
    {input}
    '''

    RESPONSE:
    '''
    {response}
    '''

    YOUR REFINEMENT:
    """)

    def __init__(self):
        super().__init__("SelfRefinementMethod")
    
    def run_for_one(self, client_query_fun: Callable,  input: str, comparator: Callable) -> str:
        prompt = self.prompt_template.format(input=input)
        output = client_query_fun(prompt, use_cache=True)

        refinement_prompt = self.self_refinement_prompt.format(input=input, response=output)
        refined_output = client_query_fun(refinement_prompt)

        return self.extract_answer(refined_output)



@register_test_time_method("MajorityVotingMethod")
class MajorityVotingMethod(VanillaPromptMethod):
    def __init__(self, n_votes: int = 3):
        self.n_votes = n_votes
        super().__init__(f"MajorityVotingMethod_{n_votes}")
    
    def run_for_one(self, client_query_fun: Callable, input: str, comparator: Callable) -> str:
        votes = []
        for _ in range(self.n_votes):
            prompt = self.prompt_template.format(input=input)
            output = client_query_fun(prompt, use_cache=True)
            votes.append(self.extract_answer(output))

        # use comparator to find the equal pairs
        cnt_votes = {}
        for vote in votes:
            unique_votes = list(cnt_votes.keys())
            is_unique = True
            for unique_vote in unique_votes:
                if comparator(vote, unique_vote):
                    cnt_votes[unique_vote] += 1
                    is_unique = False
                    break
            if is_unique:
                cnt_votes[vote] = 1

        # find the majority vote
        max_vote = max(cnt_votes.values())
        majority_vote = [vote for vote, count in cnt_votes.items() if count == max_vote]
        return majority_vote[0]