from cost_of_pass.tasks.base import register_task, map_rest_to_metadata, Task
from datasets import load_dataset, Dataset
from cost_of_pass.metrics import MathExpressionMatch

@register_task("MATH500")
class MATH500(Task):
    def __init__(self, task_name: str="MATH500", n_samples: int = -1):
        super().__init__(task_name, n_samples)

    def load_data(self) -> Dataset:
        
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        dataset = dataset["test"]
        
        dataset = dataset.rename_column("problem", "input")
        dataset = dataset.rename_column("answer", "target")
        
        dataset = map_rest_to_metadata(dataset)
        
        return dataset
    
    @property
    def comparator(self):
        return MathExpressionMatch.compare
    