from typing import Union
from cost_of_pass.metrics.base import EvaluationMetric, register_metric
from cost_of_pass.tasks.base import FullQuery

@register_metric("ExactStringMatch")
class ExactStringMatch(EvaluationMetric):
    """
    Evaluates if output matches target exactly after preprocessing.
    """

    def __init__(self):
        super().__init__(name='ExactStringMatch')
    
    @staticmethod
    def compare(output: str, query: Union[str, FullQuery]) -> bool:
        output = EvaluationMetric.preprocess(output)
        if isinstance(query, str):
            target = EvaluationMetric.preprocess(query)
        else:
            target = EvaluationMetric.preprocess(query.target)
        return output == target

@register_metric("SoftStringMatch")
class SoftStringMatch(EvaluationMetric):
    """
    Evaluates if the target exists as a substring in the output after preprocessing.
    """

    def __init__(self):
        super().__init__(name='SoftStringMatch')
    
    @staticmethod
    def compare(output: str, query: Union[str, FullQuery]) -> bool:
        output = EvaluationMetric.preprocess(output)
        if isinstance(query, str):
            target = EvaluationMetric.preprocess(query)
        else:
            target = EvaluationMetric.preprocess(query.target)
        return target in output