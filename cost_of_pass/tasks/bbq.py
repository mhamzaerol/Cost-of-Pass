from cost_of_pass.tasks.base import register_task, map_rest_to_metadata, Task
from datasets import load_dataset, concatenate_datasets, Dataset
from cost_of_pass.metrics import MultipleChoiceMatch

@register_task("BBQ")
class BBQ(Task):

    def __init__(self, task_name: str="BBQ", n_samples: int = -1, context_condition: str = None):
        self.context_condition = context_condition
        super().__init__(task_name, n_samples)

    def load_data(self) -> Dataset:
        
        dataset = concatenate_datasets(list(load_dataset("Elfsong/BBQ").values()))
        
        if self.context_condition: # for filtering ambiguous examples
            dataset = dataset.filter(lambda x: x['context_condition'] == self.context_condition) 
        
        dataset = dataset.map(lambda x: {
                **x,
                'input': f'{x["context"]} {x["question"]}\n(a) {x["ans0"]}\n(b) {x["ans1"]}\n(c) {x["ans2"]}',
                'target': f'({chr(97 + int(x["answer_label"]))})'
            },
            remove_columns=['context', 'question', 'ans0', 'ans1', 'ans2']
        )

        dataset = map_rest_to_metadata(dataset)

        return dataset
    
    @property
    def comparator(self):
        return MultipleChoiceMatch.compare