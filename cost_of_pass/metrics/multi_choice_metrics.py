from typing import Union
import re
from cost_of_pass.metrics.base import EvaluationMetric, register_metric
from cost_of_pass.tasks.base import FullQuery

@register_metric("MultipleChoiceMatch")
class MultipleChoiceMatch(EvaluationMetric):
    """
    Evaluates whether the model's output matches the correct multiple choice answer.
    Implements a prioritized strategy to extract the answer.
    """

    def __init__(self):
        super().__init__(name="MultipleChoiceMatch")

    @staticmethod
    def extract_letter_from_text(text: str) -> str:
        text = text.strip().upper()

        # Priority 1: (A)
        match = re.search(r"\(([A-F])\)", text)
        if match:
            return match.group(1)

        # Priority 2: [A]
        match = re.search(r"\[([A-F])\]", text)
        if match:
            return match.group(1)

        # Priority 3: "A"
        match = re.search(r"\"([A-F])\"", text)
        if match:
            return match.group(1)

        # Priority 4: A)
        match = re.search(r"\b([A-F])\)", text)
        if match:
            return match.group(1)

        # Priority 5: : A or - A
        match = re.search(r"[:\-]\s*([A-F])\b", text)
        if match:
            return match.group(1)

        # Priority 6: = A
        match = re.search(r"=\s*([A-F])\b", text)
        if match:
            return match.group(1)

        # Priority 7: Surrounded A (space or punctuation)
        match = re.search(r"[^A-Z]([A-F])[^A-Z]", f" {text} ")  # padding avoids edge issues
        if match:
            return match.group(1)

        # Fallback 1: Last standalone A-F
        matches = re.findall(r"\b([A-F])\b", text)
        if matches:
            return matches[-1]

        # Fallback 2: First A-F anywhere
        for ch in text:
            if 'A' <= ch <= 'F':
                return ch

        return ""

    @staticmethod
    def compare(output: str, query: Union[str, FullQuery]) -> bool:
        output = EvaluationMetric.preprocess(output, remove_punct=False, lowercase=False)
        if isinstance(query, str):
            target = EvaluationMetric.preprocess(query, remove_punct=False, lowercase=False)
        else:
            target = EvaluationMetric.preprocess(query.target, remove_punct=False, lowercase=False)
        output_letter = MultipleChoiceMatch.extract_letter_from_text(output)
        target_letter = MultipleChoiceMatch.extract_letter_from_text(target)

        return output_letter == target_letter