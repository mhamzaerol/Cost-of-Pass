from abc import ABC, abstractmethod
import string
import unicodedata
import re
from typing import Union, List, Dict
from cost_of_pass.tasks.base import FullQuery

class EvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics with a modular preprocessing pipeline.
    """

    def __init__(self, name: str = None):
        self.name = name

    @staticmethod
    @abstractmethod
    def compare(output: str, query: Union[str, FullQuery]) -> bool:
        """
        Compares the output against the query. 
        One way to do is to compare against the query.target.
        """
        pass

    def evaluate(
        self, outputs: Union[str, List[str]], queries: Union[str, FullQuery, List[Union[str, FullQuery]]]
    ) -> Dict:
        """
        Evaluates the outputs against the targets using the comparison function.
        """
        outputs = EvaluationMetric.to_list(outputs)
        queries = EvaluationMetric.to_queries(EvaluationMetric.to_list(queries))

        # Handle broadcasting if target is a single string but outputs are multiple
        if len(queries) == 1 and len(outputs) > 1:
            queries *= len(outputs)

        # Ensure matching list lengths
        if len(outputs) != len(queries):
            raise ValueError(
                "Outputs and targets must have the same length or targets must be a single string."
            )

        results = []
        for output, query in zip(outputs, queries):
            is_equivalent = self.__class__.compare(output, query)
            results.append(is_equivalent)

        correct = sum(results)
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        return {
            "results": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    # Helper functions
    @staticmethod
    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def convert_newline_to_space(text: str) -> str:
        return text.replace("\n", " ").strip()

    @staticmethod
    def convert_to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_accents(text: str) -> str:
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    @staticmethod
    def preprocess(
        text: str,
        lowercase: bool = True,
        remove_punct: bool = True,
        normalize_ws: bool = True,
        remove_acc: bool = False,
    ) -> str:
        """
        Preprocesses text based on provided flags.
        """
        if lowercase:
            text = EvaluationMetric.convert_to_lower(text)
        if remove_punct:
            text = EvaluationMetric.remove_punctuation(text)
        if normalize_ws:
            text = EvaluationMetric.normalize_whitespace(text)
        if remove_acc:
            text = EvaluationMetric.remove_accents(text)
        return text

    @staticmethod
    def to_list(data: Union[str, FullQuery, List[Union[str, FullQuery]]]) -> List[Union[str, FullQuery]]:
        """
        Converts input to list if it's a single string.
        """
        if isinstance(data, list):
            return data
        return [data]
    
    @staticmethod
    def to_queries(data: List[Union[str, FullQuery]]) -> List[FullQuery]:
        """
        Converts input to list of FullQuery objects.
        """
        return [
            FullQuery(target=entry, input="", idx=-1) if isinstance(entry, str) else entry
            for entry in data
        ]

_metric_registry = {}

def register_metric(name: str):
    def decorator(cls):
        if name in _metric_registry:
            raise ValueError(f"Metric '{name}' already registered.")
        _metric_registry[name] = cls
        return cls
    return decorator

def get_metric(name: str, *args, **kwargs):
    if name not in _metric_registry:
        raise ValueError(f"Metric '{name}' not found in registry. Available metrics: {list(_metric_registry.keys())}")
    return _metric_registry[name](*args, **kwargs)

def list_metrics() -> List[str]:
    """Return a list of all registered metric names."""
    return list(_metric_registry.keys())