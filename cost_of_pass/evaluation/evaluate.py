from typing import Union, List, Tuple, Callable
from cost_of_pass.tasks import Task, get_task
from cost_of_pass.models import Client, get_client
from cost_of_pass.models.base import GenerationLog
from cost_of_pass.evaluation.recording import FullRecord, MetricRecord, HubManager
from cost_of_pass.testtime.base import TestTimeMethod, get_test_time_method
from cost_of_pass.testtime.basic import VanillaPromptMethod
from cost_of_pass.tasks.base import FullQuery
from cost_of_pass.metrics.base import EvaluationMetric, get_metric
import concurrent.futures
import random
from cost_of_pass.evaluation.utils import group_and_trim_records

class Evaluator:
    """
    Runs and evaluates a model on a task. Can save the results to a hub. 
    """
    def __init__(
        self,
        model: Union[str, Client],
        task: Union[str, Task],
        tt_method: Union[str, TestTimeMethod],
        hub_manager: Union[None, HubManager] = None,
        model_kwargs: dict = {},
        task_kwargs: dict = {},
        tt_method_kwargs: dict = {},
        hub_manager_kwargs: dict = {},
    ):
        self.model_kwargs = model_kwargs
        self.task_kwargs = task_kwargs
        self.tt_method_kwargs = tt_method_kwargs
        self.hub_manager_kwargs = hub_manager_kwargs

        self.model = get_client(model, **model_kwargs) if isinstance(model, str) else model
        self.task = get_task(task, **task_kwargs) if isinstance(task, str) else task
        self.tt_method = get_test_time_method(tt_method, **tt_method_kwargs) if isinstance(tt_method, str) else tt_method
        self.hub_manager = HubManager(self.model, self.task, self.tt_method, **hub_manager_kwargs) if hub_manager is None else hub_manager


    def run_one_and_monitor(self, query: FullQuery, lookup_dict: Union[None, dict] = None, comparator: Union[None, Callable] = None, **kwargs) -> FullRecord:
        """
        Run a single query and monitor the run, returning a FullRecord.
        """
        def _generate_from_cache(prompts: Union[str, List[str]]) -> Union[GenerationLog, List[GenerationLog]]:
            if isinstance(prompts, str):
                prompts = [prompts]
            list_of_interest = lookup_dict[query.idx]
            choices = random.choices(list_of_interest, k=len(prompts))
            ret = [
                GenerationLog(
                    prompt=prompt,
                    response=choice.responses[-1],
                    prompt_n_tokens=choice.num_prompt_tokens,
                    response_n_tokens=choice.num_completion_tokens
                )
                for prompt, choice in zip(prompts, choices)
            ]
            if len(ret) == 1:
                return ret[0]
            return ret

        if lookup_dict is not None:
            monitored_run_func, logs = self.model.get_monitored_run_func(_generate_from_cache, **kwargs)
        else:
            monitored_run_func, logs = self.model.get_monitored_run_func(**kwargs)
        
        try:
            if comparator is None:
                comparator = self.task.comparator
            answer = self.tt_method.run_for_one(monitored_run_func, query.input, comparator)
            completed = True
        except Exception as e:
            answer = f'<incomplete>{str(e)}</incomplete>'
            completed = False

        return FullRecord(
            model_name=self.model.model_name,
            task_name=self.task.task_name,
            tt_method_name=self.tt_method.method_name,
            input_idx=query.idx,
            input=query.input,
            num_input_tokens=self.model.count_all_tokens(query.input),
            prompts=GenerationLog.all_prompts(logs),
            num_prompt_tokens=GenerationLog.all_prompt_n_tokens(logs),
            responses=GenerationLog.all_responses(logs),
            num_completion_tokens=GenerationLog.all_response_n_tokens(logs),
            answer=answer,
            num_answer_tokens=self.model.count_all_tokens(answer),
            target=query.target,
            cost_per_prompt_token=self.model.cost_per_prompt_token,
            cost_per_completion_token=self.model.cost_per_completion_token,
            completed=completed,
        )
    
    def run_parallel_and_monitor(self, queries: List[FullQuery], workers: int = 64, comparator: Union[None, Callable] = None, **kwargs) -> List[FullRecord]:
        """Run multiple queries in parallel using a thread pool with progress monitoring."""

        print(f"Running {len(queries)} queries in parallel using {workers} workers.")
        print(f'with the kwargs: {kwargs}')
        
        if self.tt_method.method_name != 'VanillaPromptMethod':
            vanilla_hub_manager = HubManager(self.model, self.task, VanillaPromptMethod(), **self.hub_manager_kwargs)
            lookup_dict = {}
            for record in list(vanilla_hub_manager.full_records):
                if record.input_idx not in lookup_dict:
                    lookup_dict[record.input_idx] = []
                lookup_dict[record.input_idx].append(record)
        else:
            lookup_dict = None

        if 'MajorityVotingMethod' in self.tt_method.method_name:
            return [self.run_one_and_monitor(query, lookup_dict, comparator, **kwargs) for query in queries]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.run_one_and_monitor, query, lookup_dict, comparator, **kwargs) for query in queries]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    def rerun_bad_list_records(self, full_records: List[FullRecord], **kwargs) -> List[FullRecord]:
        """
        reruns the records that could not have been evaluated successfully.
        """
        good_records = [r for r in full_records if r.completed]
        reeval_bad_queries = [r.as_query for r in full_records if not r.completed]
        print(f"There are {len(reeval_bad_queries)} bad records to reevaluate.")
        full_records = good_records + self.run_parallel_and_monitor(reeval_bad_queries, **kwargs)

        return full_records

    def rerun_bad_hub_records(self, **kwargs):
        """
        reruns the records that could not have been evaluated successfully.
        """
        full_records = self.load_full_records_from_hub()
        full_records = self.rerun_bad_list_records(full_records, **kwargs)
        self.overwrite_full_records_to_hub(full_records)

    def run(self, n_runs: int = 1, use_hub_cache: bool = True, update_hub: bool = True, append: bool = False, **kwargs) -> Tuple[List[FullRecord], bool]:
        """
        Runs the model on the task, running each question n_runs times.
        """
        if (use_hub_cache or update_hub) and self.hub_manager is None:
            raise ValueError("Cannot use hub cache or update hub without a hub manager!")
    
        if append and use_hub_cache:
            raise ValueError("Using hub cache and append together is not supported!")

        full_records_changed = False

        if use_hub_cache:
            existing_n_runs = self.get_existing_n_runs_full_records()
            full_records = self.load_full_records_from_hub()
        else:
            existing_n_runs = 0
            full_records = []
        
        # Calculate additional runs needed
        need_runs = max(0, n_runs - existing_n_runs)
        queries = self.task.get_queries(need_runs)

        # Run new evaluations in parallel and append the results
        new_records = self.run_parallel_and_monitor(queries, **kwargs)
        full_records += new_records

        # Process and filter complete records
        ret_records, rem_records = group_and_trim_records(full_records, n_runs)

        n_was_bad = sum(1 for r in ret_records if not r.completed)
        ret_records = self.rerun_bad_list_records(ret_records, **kwargs)
        n_bad_now = sum(1 for r in ret_records if not r.completed)

        full_records = ret_records + rem_records
        full_records_changed = (n_was_bad != n_bad_now) or (need_runs > 0)

        if n_bad_now > 0:
            print(f"WARNING: {n_bad_now} RECORDS ARE STILL BAD.")
            print("For reevaluation of the returned records, use rerun_bad_list_records method!")
            print("For reevaluation of the records in the hub, use rerun_bad_hub_records method!")

        # Update hub if required and if records have changed
        if update_hub and full_records_changed:
            if append:
                self.append_full_records_to_hub(full_records)
            else:
                self.overwrite_full_records_to_hub(full_records)
        
        return ret_records, (n_bad_now > 0)

    def evaluate(self, metric: Union[str, EvaluationMetric], n_runs: int = 1, use_hub_cache: bool = True, update_hub: bool = True, append: bool = False, ignore_bad: bool = False, **kwargs) -> List[MetricRecord]:
        """
        Evaluates the model on the task using the specified metric, running each question n_runs times.
        """
        # Get the metric object
        if isinstance(metric, str):
            metric = get_metric(metric)

        if (use_hub_cache or update_hub) and self.hub_manager is None:
            raise ValueError("Cannot use hub cache or update hub without a hub manager!")
        
        if append and use_hub_cache:
            raise ValueError("Using hub cache and append together is not supported!")
        
        metric_records_changed = False

        if use_hub_cache:
            existing_n_runs = self.get_existing_n_runs_metric_records(metric)
        else:
            existing_n_runs = 0

        if existing_n_runs >= n_runs:
            existing_metric_records = self.load_metric_records_from_hub(metric)
            metric_records, _ = group_and_trim_records(existing_metric_records, n_runs)
            has_bad = False
        else:
            records, has_bad = self.run(n_runs, use_hub_cache, update_hub, append, comparator=metric.__class__.compare, **kwargs)
            if has_bad:
                if ignore_bad:
                    cnt_bad = sum(1 for r in records if not r.completed)
                    print(f"!!!WARNING WARNING WARNING!!!: {cnt_bad} RECORDS ARE BAD. Ignoring (excluding) them for evaluation!!")
                    records = [r for r in records if r.completed]
                else:
                    raise ValueError("There are incomplete records. Please reevaluate / rerun!")
        
            outputs = [r.answer for r in records]
            queries = [r.as_full_query for r in records]

            eval_results = metric.evaluate(outputs, queries)
            print(f'evaluated {len(records)} records using metric {metric.name}')
            print('mean score:', eval_results['accuracy'])
            print('total correct:', eval_results['correct'])
            print('total:', eval_results)

            metric_records = [
                MetricRecord.from_base(
                    record.as_base(), 
                    metric_name=metric.name, 
                    metric_score=float(metric_score) # this returns T/F, converting to float
                )
                for metric_score, record in zip(eval_results['results'], records)
            ]
            metric_records_changed = True

        if update_hub and metric_records_changed:
            if has_bad:
                print("Hub records are not updated due to bad records!")
            elif append:
                self.append_metric_records_to_hub(metric, metric_records)
            else:
                self.overwrite_metric_records_to_hub(metric, metric_records)

        return metric_records
    
    def append_full_records_to_hub(self, records: List[FullRecord]):
        self.hub_manager.save_full_records(records, overwrite=False)
    
    def append_metric_records_to_hub(self, metric: Union[str, EvaluationMetric], records: List[MetricRecord]):
        self.hub_manager.save_metric_records(metric, records, overwrite=False)

    def overwrite_full_records_to_hub(self, records: List[FullRecord]):
        self.hub_manager.save_full_records(records, overwrite=True)
    
    def overwrite_metric_records_to_hub(self, metric: Union[str, EvaluationMetric], records: List[MetricRecord]):
        self.hub_manager.save_metric_records(metric, records, overwrite=True)

    def load_full_records_from_hub(self):
        return self.hub_manager.full_records
    
    def load_metric_records_from_hub(self, metric: Union[str, EvaluationMetric]):
        return self.hub_manager.metric_records(metric)

    def get_existing_n_runs_full_records(self):
        metadata = self.hub_manager.full_records_metadata
        return metadata.n_runs
    
    def get_existing_n_runs_metric_records(self, metric: Union[str, EvaluationMetric]):
        metadata = self.hub_manager.metric_records_metadata(metric)
        return metadata.n_runs