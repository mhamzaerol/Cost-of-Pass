from typing import List, Union, Dict
from collections import defaultdict

from cost_of_pass.tasks.base import Task
from cost_of_pass.metrics.base import EvaluationMetric
from cost_of_pass.testtime.base import TestTimeMethod
from cost_of_pass.models import Client, get_client
from cost_of_pass.evaluation.recording import MetricRecord
from cost_of_pass.tasks import get_task
from cost_of_pass.metrics import get_metric
from cost_of_pass.evaluation import Evaluator
from cost_of_pass.testtime import get_test_time_method
from cost_of_pass.evaluation.utils import group_records_by_input_idx, expected_value

class ModelFamily:
    """
    Defines a model family consisting of pairs of models and inference time methods.
    """
    def __init__(self, name: str, other_model_family: 'ModelFamily' = None):
        self.name = name
        self.models = []
        if other_model_family is not None:
            for (m, t, r) in other_model_family.models:
                if isinstance(m, str):
                    m = get_client(m)
                if isinstance(t, str):
                    t = get_test_time_method(t)
                self.models.append((m, t, r))

    def del_model(self, model: Union[str, Client], tt_method: Union[str, TestTimeMethod] = None) -> 'ModelFamily':
        if isinstance(model, str):
            model = get_client(model)
        if isinstance(tt_method, str):
            tt_method = get_test_time_method(tt_method)
        self.models = [(m, t, r) for m, t, r in self.models if m.model_name != model.model_name or t.method_name != tt_method.method_name]
        return self
    
    def add_model(self, model: Union[str, Client], tt_method: Union[str, TestTimeMethod], runtime_kwargs: dict = {}, **kwargs) -> 'ModelFamily':
        if isinstance(model, str):
            model = get_client(model)
        if isinstance(tt_method, str):
            tt_method = get_test_time_method(tt_method)
        self.models.append((model, tt_method, runtime_kwargs))
        return self

class FrontierCostofPass:
    """
    Computes the frontier cost of pass of a model family for a given task.
    """
    INF = float('inf')

    def __init__(self, task: Union[str, Task], baseline_cop: float, metric: Union[str, EvaluationMetric], **kwargs):
        self.task = get_task(task, **kwargs['task_kwargs']) if isinstance(task, str) else task
        self.baseline_cop = baseline_cop
        self.metric = get_metric(metric, **kwargs['metric_kwargs']) if isinstance(metric, str) else metric
    
    def cost_of_pass(self, records: List[MetricRecord]) -> float:
        E_cost = expected_value([record.cost for record in records])
        E_performance = expected_value([record.performance for record in records])
        
        if E_performance == 0:
            return self.INF
        
        return E_cost / E_performance

    def estimate_each_problem(self, model_family: ModelFamily, n_runs: int = 1, exclude_baseline: bool = False, ignore_bad_records: bool = False, **kwargs) -> Dict[int, float]:
        cost_of_passes = defaultdict(list)
        for model, tt_method, runtime_kwargs in model_family.models:
            evaluator = Evaluator(
                model, self.task, tt_method, 
                kwargs.pop("hub_manager", None),
                kwargs.pop("model_kwargs", {}),
                kwargs.pop("task_kwargs", {}),
                kwargs.pop("tt_method_kwargs", {}),
                kwargs.pop("hub_manager_kwargs", {}),
            )
            metric_records = evaluator.evaluate(self.metric, n_runs=n_runs, ignore_bad=ignore_bad_records, **kwargs, **runtime_kwargs)
            good_group, _, has_bad = group_records_by_input_idx(metric_records)
            if has_bad:
                if ignore_bad_records:
                    print(f"Warning: Some records are not completed for the model {evaluator.model.model_name} with test time method {evaluator.test_time_method.name}.")
                    print("You chose to ignore them, so we will proceed.")
                else:
                    raise ValueError(f"Some records are not completed for the model {evaluator.model.model_name} with test time method {evaluator.test_time_method.name}.")
            for input_idx, records in good_group.items():
                cost_of_passes[input_idx].append(self.cost_of_pass(records))
        
        if not exclude_baseline:
            if len(model_family.models) == 0:
                cost_of_passes[-1] = [self.baseline_cop]
            else:
                for input_idx in list(cost_of_passes.keys()):
                    cost_of_passes[input_idx].append(self.baseline_cop)
        
        frontier_cost_of_passes = defaultdict(float)
        for input_idx in cost_of_passes:
            min_cost_of_pass = min(cost_of_passes[input_idx])
            frontier_cost_of_passes[input_idx] = min_cost_of_pass
        
        return frontier_cost_of_passes

    def estimate_task(self, model_family: ModelFamily, n_runs: int = 1, exclude_baseline: bool = False, ignore_bad_records: bool = False, **kwargs) -> float:
        frontier_cost_of_passes = self.estimate_each_problem(model_family, n_runs, exclude_baseline, ignore_bad_records, **kwargs)
        if any(cost == self.INF for cost in frontier_cost_of_passes.values()):
            raise ValueError("Under this settings, the frontier cost of pass is inifinite for some problems, thus we cannot estimate the frontier cost of pass for the task. For details, consider running the estimate_each_problem method with the same settings.")
        
        return expected_value(list(frontier_cost_of_passes.values()))