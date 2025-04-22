from cost_of_pass.tasks.base import Task, register_task
from datasets import load_from_disk # type: ignore
from pathlib import Path
from cost_of_pass.metrics import MathExpressionMatch


@register_task("AIME_2024")
class AIME_2024(Task):
    def __init__(self, task_name: str="AIME_2024", n_samples: int=-1):
        super().__init__(task_name, n_samples)

    def load_data(self):
        dataset = load_from_disk(Path(__file__).parent / "data" / "AIME_2024")
        return dataset
    
    @property
    def comparator(self):
        return MathExpressionMatch.compare