from cost_of_pass.tasks.base import register_task, map_rest_to_metadata, Task
from datasets import load_dataset, Dataset
from cost_of_pass.metrics import MathExpressionMatch

@register_task("GSM8K")
class GSM8K(Task):

    def __init__(self, task_name: str="GSM8K", n_samples: int = -1):
        super().__init__(task_name, n_samples)

    def load_data(self) -> Dataset:
        # Load the dataset from the Hugging Face hub
        dataset = load_dataset("openai/gsm8k", "main")
        dataset = dataset["test"] 

        # Rename the column 'question' to 'input' and 'answer' to 'target'
        dataset = dataset.rename_column("question", "input")
        dataset = dataset.rename_column("answer", "target")

        # Extract the ground truth from 'target' by splitting on '####'
        def extract_ground_truth(example):
            parts = example['target'].split('####')
            ground_truth = parts[-1].strip().lower()
            example['target'] = ground_truth
            return example

        dataset = dataset.map(extract_ground_truth)

        dataset = map_rest_to_metadata(dataset)

        return dataset

    @property
    def comparator(self):
        return MathExpressionMatch.compare