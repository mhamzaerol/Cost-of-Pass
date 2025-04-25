<div align="center">

# Cost-of-Pass: An Economic Framework for Evaluating Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.13359-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2504.13359)
[![Benchmark](https://img.shields.io/badge/Benchmark-HuggingFace-ffcc00.svg?style=flat&logo=huggingface)](https://huggingface.co/CostOfPass)
</div>


## Index
- [Overview](#overview)
- [Examples](#examples)
- [Setup](#setup)
- [Detailed Usage](#detailed-usage)
- [Citation](#citation)

## Overview
<div align="center">
    <img src="framework_overview.png" alt="Framework Overview" style="width: 100%;"/>
</div> <br>
The widespread adoption of AI systems in the economy hinges on their ability to generate economic value that outweighs their inference costs. Evaluating this tradeoff requires metrics that account for both performance and costs.

We propose a framework grounded in production theory for evaluating language models by combining accuracy and inference cost. We introduce **Cost-of-Pass**, the expected monetary cost of generating a correct solution. We then define the **Frontier Cost-of-Pass** as the minimum Cost-of-Pass achievable across available models or *the human expert*, using the approximate cost of hiring an expert.

With our framework, we quantify the economic benefit that language models provide over an human expert baseline. We then track the evolution of cost-efficiency over the past year across different task types, evaluate the essentialness of various model innovations, and assess the economic value of common inference-time techniques.

Our findings point to clear trends in cost-efficiency across model classes and task types, reflecting the broader dynamics of innovation in the field. These patterns, and the shifts we've observed over time, offer a window into how economic value is increasingly shaped by model-level advances rather than surface-level improvements.

## Examples
The following examples demonstrate how to use various components of our evaluation framework. 
### Creating a Client
```python
from cost_of_pass import get_client, list_clients, SamplingArgs

# List the clients, you should get ~everything available in litellm.
print("Available Clients:", list_clients())

# Retrieve a client (adjust key as needed, e.g., 'gpt-4o')
client = get_client("gpt-4o")
```

### Creating a Task
```python
from cost_of_pass import get_task, list_tasks

# List available tasks
print("Available Tasks:", list_tasks())

# Get a specific task
task = get_task("AIME_2024")
print("Task queries:", task.get_queries())
```

### Creating an Inference Time Technique
```python
from cost_of_pass import get_test_time_method, list_test_time_methods

# List available test-time methods
print("Available Test-Time Methods:", list_test_time_methods())

# Get a specific test-time method
test_time_method = get_test_time_method("VanillaPromptMethod")
```

### Measuring with a Metric
```python
from cost_of_pass import get_metric, list_metrics

# List available metrics
print("Available Metrics:", list_metrics())

# Get a specific metric
metric = get_metric("MathExpressionMatch")
```

### Computing Frontier Cost-of-Pass
Note that the passed kwargs for the hub_manager does not support pushing to the hub yet by different users. Therefore, for your own new experiments, you can create a new dataset in your organization / account, and set the `org_id` and `repo_id` accordingly (organization name / your username, and the dataset name). 
```python
from cost_of_pass import ModelFamily, FrontierCostofPass

# Create a FrontierCostofPass object
frontier_cop = FrontierCostofPass(
    task=task,
    baseline_cop=20.0, # Human expert cost-of-pass
    metric=metric,
    hub_manager_kwargs = { # details for the record management
        'org_id': 'CostOfPass', # The organization ID to use for the hub
        'repo_id': 'benchmark', # The repo ID to use for the hub
        'token': '...', # HF token (in case you want to use a different token than the one in the .env file)
    }
)

# Create a model family
model_family = ModelFamily(name="Example Model Family")

# Add models to the family
model_family = model_family.add_model(
    model=client,
    tt_method=test_time_method,
)

# Estimate the frontier cost-of-pass for the task under the model family
print(
    frontier_cop.estimate_task(
        model_family=model_family,
        n_runs=8,
    )
)
```

## Setup
Our repository integrates [LiteLLM](https://github.com/BerriAI/litellm/tree/main) for querying language models and uses [Hugging Face](https://huggingface.co/CostOfPass) to store and share evaluation results. Follow these steps to setup the repository:

1. **Clone the repository** 
2. **Create a Python environment:** Use a tool like `Conda` or `venv`. For instance:
```bash
conda create -n cost_of_pass python=3.11
conda activate cost_of_pass
```
3. **Install the package:** Run the following command from the repository root:
```bash
pip install -e .
```
4. **Configure API access:** To query models via LiteLLM (or your custom API) and push results to the Hugging Face Hub (of yours / ours), create a `.env` file with the necessary credentials:
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
# Add other relevant keys as needed
```

## Detailed Usage
### Analysis
Please check the `analysis/analysis.ipynb` file for the notebook on reproducing the analyses in the paper. 
### Passing Custom Args
You can pass custom arguments to tasks, inference time methods, and evaluations. 
<details>
<summary>Show example code</summary>

```python
# Modifying task name and number of samples to work with
task = get_task("GPQA_Diamond", task_name="My_GPQA_Diamond_Task", n_samples=128)

# Some inference time methods may require additional arguments
tt_method = get_test_time_method("MajorityVotingMethod", n_votes=3)

# An alternative way of initializing the FrontierCostofPass object
frontier_cop = FrontierCostofPass(
    task="GPQA_Diamond",
    task_kwargs={"task_name": "My_GPQA_Diamond_Task", "n_samples": 128},
    baseline_cop=58.0,  # Human expert cost-of-pass
    metric="ExactStringMatch",
    metric_kwargs={},
)

# Passing custom runtime arguments when the model is run
model_family = ModelFamily(name="Example Model Family")
model_family = model_family.add_model(
    model=client,
    tt_method=test_time_method,
    runtime_kwargs={
        'sampling_args': SamplingArgs(temperature=0.9, top_p=1.0, max_tokens=2048), # Sampling arguments for the model
        'max_attempts': 5,  # Number of attempts to generate a response
        'sleep_time': 2, # How long to sleep after each failed attempt
        'sleep_multiplier': 1.5, # How much to multiply the sleep time after each failed attempt
        'timeout': 10 # Timeout for the request (i.e. how many seconds should we wait for a response before giving up)
    }
)

# Passing different args to the frontier_cop estimation
frontier_cop.estimate_task(
    model_family=model_family,
    n_runs=8,
    exclude_baseline=True, # Exclude the human expert from the estimation: LM Frontier CoP
    ignore_bad_records=True, # Ignore bad records (e.g. when evaluation fails for a model / problem)
    # kwargs for evaluation
    use_hub_cache=True, # Use the cached evaluation records from the hub (if available)
    update_hub=True, # Update the hub with the newly generated evaluation records
    append=True, # Should append or overwrite the existing records in the hub (if updating)
    workers=64, # Number of workers to use for parallel API calls
)
```

</details>

### Custom Benchmark
If you would like to create a custom benchmark, using different models, tasks, metrics, inference time methods, or different cost / performance estimations, please check these information and follow the instructions:
#### Implementing a Custom Task
Each custom task should inherit from the `Task` class implemented in `tasks/base.py`, and should support the implementation for the `load_data` and `comparator` methods. The former is responsible for loading the data into a common format (a dictionary with keys `input`, `target` and optionally `metadata`), the latter provides a comparator function to be used in an inference-time method requiring such comparison (e.g. `MajorityVotingMethod`). Moreover, the custom task's `__init__` method can take alternative arguments that could affect the behavior in these methods (e.g. a specific subset type of the dataset could be passed as an argument). When such a custom task is imported inside the `tasks/__init__.py` file, it will be automatically registered and available for use with the `get_task` function.
#### Implementing a Custom Model
Even though many models are already available through `LiteLLM` (implemented in `clients/litellm.py`), you can still implement custom model API call interfaces by inheriting from `base.py/Client` class. Any custom model should implement the following methods:
- `generate_one`: Given a prompt, sampling arguments, and other parameters, this should support the generation of a single response from the model, and return a generation log containing the prompt, response, and token counts.
- `count_tokens`: Given a text, this should return how many tokens the model would use to represent the text.
- `cost_per_prompt_token`: This should return the cost per prompt token for the model.
- `cost_per_completion_token`: This should return the cost per completion token for the model.
Importing the custom model inside the `models/__init__.py` file will automatically register it and make it available for use with the `get_client` function.
#### Implementing a Custom Inference Time Method
Currently, we have implemented basic inference time methods such as `VanillaPromptMethod`, `SelfRefinementMethod` and `MajorityVotingMethod`. You can implement alternative methods by inheriting from the `base.py/TestTimeMethod` class. Any such custom method should implement the `run_for_one` method, which takes an input query, a client query function (that is monitored (for token consumption) whenever the method makes a call) and a comparator function that can compare two of the generated outputs (if needed). The output of this method should be a string, indicating the final output of the method. Similarly, importing the custom method inside the `testtime/__init__.py` file will automatically register it and make it available for use with the `get_test_time_method` function.
#### Implementing a Custom Metric
Any new custom metric should inherit from the `base.py/EvaluationMetric` class, and implement the `compare` static method. This method takes an output string and a query object, and scores the output (right now binary) based on its satisfactoriness. Importing the custom metric inside the `metrics/__init__.py` file will automatically register it and make it available for use with the `get_metric` function.
#### Evaluation
In case you may want to generate records of running a model / inference-time method on a task before evaluating with a metric or using in a cost-of-pass based analysis, you can use the `Evaluator` class under the `evaluation/evaluate.py` file:
```python
from cost_of_pass import Evaluator

evaluator = Evaluator(
    model=..., 
    task=..., 
    tt_method=...,
    hub_manager_kwargs = {
        'org_id': ...,
        'repo_id': ...,
    }
)
```
This class takes a model, task and an inference-time method, and supports running for queries with multiple arguments (check the `run` method). This class supports saving or loading the generated records to/from a Hugging Face Hub (e.g. check our benchmark repo [here](https://huggingface.co/CostOfPass/benchmark)), by specifying arguments in the `__init__` method inside the `hub_manager_kwargs` which are passed to the `HubManager` class (important ones: `org_id`, `repo_id` and `token` (if not provided through `.env` file)). Evaluator's `run` method supports loading / saving records from / to hub (`use_hub_cache`, `update_hub`) and requests the user on how many times a model should be run per query (`n_runs`): 
```python
full_records, has_bad = evaluator.run(
    n_runs=8, # Number of runs per query
    use_hub_cache=True, # Use the cached records from the hub (if available)
    update_hub=True, # Update the hub with the newly generated records
    workers=64, # Number of workers to use for parallel API calls
)
```
These records are by default saved / loaded to / from the hub, as `FullRecord` object defined under the `recording.py` file (determined by the `use_hub_cache` and `update_hub` arguments). In case a user wants to evaluate the generated records (taken from hub or generated on-the-fly), the `evaluate` method of the `Evaluator` class can be used: 
```python
metric_records = evaluator.evaluate(
    metric=..., # The metric to use for evaluation
    use_hub_cache=True, # Use the cached records from the hub (if available)
    update_hub=True, # Update the hub with the newly generated records
    workers=64, # Number of workers to use for parallel API calls
)
```
This method takes a metric and similar arguments as the `run` method, and generates / loads records (according to the availability / arguments passed), evaluates them, and by default saves them to the hub as a `MetricRecord` object. It returns a list of `MetricRecord` objects. Note that we currently do not support passing custom full_records into the `evaluate` method, instead this method either loads saved records from the hub or generates new ones. 
Given these, a user may firstly generate `FullRecord` objects, store them in the hub, then evaluate them with a metric and store the `MetricRecord` objects in the hub. Finally, use them to estimate the frontier cost-of-pass in their analyses, in such separable steps (automatically supported by our codebase). Notably, this (using hub as a cache) saves API calls / model generations, and allows for reproducible results.
#### Extension Directions
One might want to improve the `HubManager` implementation to make more abstract for supporting various record storage systems (e.g. local storage, different cloud storage systems etc.) for custom purposes. 
## Citation
If you find our work useful, please consider citing:
```bibtex
@misc{erol2025costofpass,
      title={Cost-of-Pass: An Economic Framework for Evaluating Language Models}, 
      author={Mehmet Hamza Erol and Batu El and Mirac Suzgun and Mert Yuksekgonul and James Zou},
      year={2025},
      eprint={2504.13359},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.13359}, 
}
```
