from abc import ABC, abstractmethod
from typing import Union, List, Any, Callable
from datasets import Dataset
from dataclasses import dataclass
from dataclasses import fields

@dataclass
class BaseQuery:
    input: str
    target: str
    idx: int

@dataclass
class FullQuery(BaseQuery):
    metadata: Any = None # TODO: set a standard (eg dict)

    @classmethod
    def from_base(cls, base: BaseQuery, **kwargs) -> "FullQuery":
        base_field_names = {f.name for f in fields(BaseQuery)}
        base_data = {name: getattr(base, name) for name in base_field_names}
        base_data.update(kwargs)
        return cls(**base_data)

class Task(ABC):
    """
    Base class for all tasks.
    Each task should implement its own:
        - load_data method:
            - Loads the dataset (from any source (local files, Hugging Face, etc.))
            - Needs to format it into a dictionary with the keys: 'input', 'target' and 'metadata' (optional)
        - comparator property
            - Returns a callable that takes two strings (outputs) and checks their equivalence
            - So far, only used for the majority voting
    Check example task implementations under this directory, to see various ways of constructing a task.
    """
    task_name: str
    n_samples: int = -1
    dataset: Dataset = None

    def __init__(self, task_name: str, n_samples: int = -1):
        self.task_name = task_name
        self.__post_init__(n_samples)

    def __post_init__(self, n_samples: int):
        self.dataset = self.load_data()
        self._validate_dataset()
        self.set_indices(n_samples)

        print(f"Loaded task '{self.task_name}'.")

    @abstractmethod
    def load_data(self) -> Dataset:
        """Load and return the dataset."""
        pass

    def set_indices(self, n_samples: int):
        if n_samples > 0:
            self._indices = get_k_random_indices(n_samples, len(self.dataset))
        else:
            self._indices = list(range(len(self.dataset)))
        print(f"Using {len(self._indices)} samples with params: n_samples={n_samples}")

    def _validate_dataset(self):
        # check that the dataset has the required keys
        required_keys = ['input', 'target']
        missing = [k for k in required_keys if k not in self.dataset.column_names]
        if missing:
            raise ValueError(f"Dataset is missing keys: {missing}")
        
        # check that all the cells under input / target are strings
        for key in required_keys:
            if not all(isinstance(x, str) for x in self.dataset[key]):
                raise ValueError(f"Dataset key '{key}' must contain only strings.")

    def get_queries(self, n_times: int = 1) -> List[BaseQuery]:
        return [
            BaseQuery(
                input=self.dataset['input'][i],
                target=self.dataset['target'][i],
                idx=i
            )
            for _ in range(n_times)
            for i in self.indices
        ]
    
    def get_full_queries(self, n_times: int = 1) -> List[FullQuery]:
        queries = self.get_queries(n_times)
        return [
            FullQuery.from_base(
                query,
                metadata=self.dataset['metadata'][query.idx]
            )
            for query in queries
        ]

    @property
    @abstractmethod
    def comparator(self) -> Callable:
        pass

    @property
    def inputs(self) -> List[str]:
        return [
            self.dataset['input'][i] for i in self.indices
        ]

    @property
    def targets(self) -> List[str]:
        return [
            self.dataset['target'][i] for i in self.indices
        ]
    
    @property
    def indices(self) -> List[int]:
        return self._indices[:]
    
    @property
    def metadata(self) -> Union[List[Any], None]:
        if 'metadata' in self.dataset.column_names:
            return [
                self.dataset['metadata'][i] for i in self.indices
            ]
        return None

def map_rest_to_metadata(dataset: Dataset, retain_keys: List[str] = ['input', 'target']) -> Dataset:
    modified_dataset = dataset.map(lambda x: {
        **{k: v for k, v in x.items() if k in retain_keys},
        "metadata": {k: v for k, v in x.items() if k not in retain_keys}
    })
    return modified_dataset

def get_k_random_indices(k: int, n: int, seed: int = 102217) -> List[int]:

    if k > n:
        k = n

    def _lcg_rand(state: int) -> int:
        A = 279470273
        M = 2**32
        return (A * state) % M

    arr = list(range(n))
    
    state = seed
    for i in range(n-1, 0, -1):
        state = _lcg_rand(state)
        j = state % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]

    return arr[:k]

_task_registry = {}

def register_task(name: str):
    def decorator(cls):
        if name in _task_registry:
            raise ValueError(f"Task '{name}' already registered.")
        _task_registry[name] = cls
        return cls
    return decorator

def get_task(name: str, *args, **kwargs):
    if name not in _task_registry:
        raise ValueError(f"Task '{name}' not found in registry. Available tasks: {list(_task_registry.keys())}")
    return _task_registry[name](*args, **kwargs)

def list_tasks() -> List[str]:
    """Return a list of all registered task names."""
    return list(_task_registry.keys())

