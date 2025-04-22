from typing import Union
from cost_of_pass.metrics.base import EvaluationMetric, register_metric
from cost_of_pass.tasks.base import FullQuery
from math_verify import parse, verify

@register_metric("MathExpressionMatch")
class MathExpressionMatch(EvaluationMetric):
    """
    Evaluates if the output matches the target mathematically after preprocessing.
    """

    def __init__(self):
        super().__init__(name='MathExpressionMatch')

    @staticmethod
    def make_math_exp(expression: str) -> str:
        # Implement any necessary preprocessing steps here
        if not expression.startswith("$") and not expression.endswith("$"):
            expression = f"${expression}$"
        return expression
    
    @staticmethod
    def compare(output: str, query: Union[str, FullQuery]) -> bool:
        output = EvaluationMetric.preprocess(output, remove_punct=False)
        if isinstance(query, str):
            target = EvaluationMetric.preprocess(query, remove_punct=False)
        else:
            target = EvaluationMetric.preprocess(query.target, remove_punct=False)
        math_exp_output = MathExpressionMatch.make_math_exp(output)
        math_exp_target = MathExpressionMatch.make_math_exp(target)
        try:
            # Parse the expressions using Math-Verify
            gold = parse(math_exp_target)
            answer = parse(math_exp_output)
            # Verify equivalence
            is_equivalent = verify(gold, answer)
        except Exception as e:
            # Handle parsing or verification errors
            print(f"Error: {e}")
            is_equivalent = False
        return is_equivalent
