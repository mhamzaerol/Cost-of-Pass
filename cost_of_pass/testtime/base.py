from abc import ABC, abstractmethod
from typing import List, Callable

class TestTimeMethod(ABC):
    def __init__(self, method_name: str):
        self.method_name = method_name

    @abstractmethod
    def run_for_one(self, input: str, client_query_fun: Callable, comparator: Callable = None) -> str:
        """
        An inference time method is characterized by the following parameters:
            input (str): The input to the test time method.
            client_query_fun (Callable): A function that takes a prompt and returns the model's response.
            comparator (Callable): A function that takes two strings and compares their equality.
        Workflow: 
            The method utilizes the client_query_fun (with a modification of the input (e.g. prompting)) to get the model's response. 
            Specifically for the majority voting, this method supports taking a comparator that enables assessing equivalence between different model answers (e.g. in case of multiple expressions of the same answer).
            Ideally, all the calls to the client_query_fun are monitored (prompts, responses etc. are logged externally).
        Returns: 
            The final response from the test time method.
        For a more clear understanding, please check the example implementations under the basic.py file!
        """
        pass


_test_time_method_registry = {}

def register_test_time_method(name: str):
    def decorator(cls):
        if name in _test_time_method_registry:
            raise ValueError(f"test_time_method '{name}' already registered.")
        _test_time_method_registry[name] = cls
        return cls
    return decorator

def get_test_time_method(name: str, *args, **kwargs):
    if name not in _test_time_method_registry:
        raise ValueError(f"test_time_method '{name}' not found in registry. Available test_time_methods: {list(_test_time_method_registry.keys())}")
    return _test_time_method_registry[name](*args, **kwargs)

def list_test_time_methods() -> List[str]:
    """Return a list of all registered test_time_method names."""
    return list(_test_time_method_registry.keys())

