from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple, Any
from dataclasses import dataclass
import concurrent.futures
import asyncio
from time import sleep

@dataclass
class SamplingArgs:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = None

@dataclass
class GenerationLog:
    prompt: str
    response: str
    prompt_n_tokens: int
    response_n_tokens: int

    @staticmethod
    def all_prompts(logs: Union['GenerationLog', List['GenerationLog']]) -> List[str]:
        """Return all prompts from the logs."""
        if isinstance(logs, GenerationLog):
            return [logs.prompt]
        return [log.prompt for log in logs]
    
    @staticmethod
    def all_responses(logs: Union['GenerationLog', List['GenerationLog']]) -> List[str]:
        """Return all responses from the logs."""
        if isinstance(logs, GenerationLog):
            return [logs.response]
        return [log.response for log in logs]
    
    @staticmethod
    def all_prompt_n_tokens(logs: Union['GenerationLog', List['GenerationLog']]) -> int:
        """Return the total number of prompt tokens from the logs."""
        if isinstance(logs, GenerationLog):
            return logs.prompt_n_tokens
        return sum(log.prompt_n_tokens for log in logs)
    
    @staticmethod
    def all_response_n_tokens(logs: Union['GenerationLog', List['GenerationLog']]) -> int:
        """Return the total number of response tokens from the logs."""
        if isinstance(logs, GenerationLog):
            return logs.response_n_tokens
        return sum(log.response_n_tokens for log in logs)

class Client(ABC):
    """
    Base class for all clients. All clients need to implement the following methods:
    - generate_one: Generate a generation log for a single prompt.
    - count_tokens: Return the number of tokens the model would use for a given text.
    - cost_per_prompt_token: Return the cost the model would charge for a prompt token.
    - cost_per_completion_token: Return the cost the model would charge for a completion token.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_one(self, prompt: str, sampling_args: SamplingArgs = None, **kwargs) -> GenerationLog:
        """Generate a single response from a given client. All models need to implement this."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the text."""
        pass

    @property
    @abstractmethod
    def cost_per_prompt_token(self) -> float:
        """Return the cost per token for a prompt."""
        pass

    @property
    @abstractmethod
    def cost_per_completion_token(self) -> float:
        """Return the cost per token for a completion."""
        pass
    
    def __call__(self, prompts: Union[str, List[str]], sampling_args: SamplingArgs = None, **kwargs) -> Union[GenerationLog, List[GenerationLog]]:
        """Generate responses from a given client."""
        return self.generate(prompts, sampling_args, **kwargs)

    def count_all_tokens(self, texts: Union[str, List[str]]) -> int:
        """Return the total number of tokens in the text or list of texts."""
        if isinstance(texts, str):
            return self.count_tokens(texts)
        return sum(self.count_tokens(text) for text in texts)

    def generate_one_resilient(self, prompt: str, sampling_args: SamplingArgs = None, **kwargs) -> GenerationLog:
        """Generate a single response from a given client resiliently. Handles errors and retries."""
        n_attempts = 0
        max_attempts = kwargs.pop("max_attempts", 1)
        error = None
        sleep_time = kwargs.pop("sleep_time", 2)
        sleep_multiplier = kwargs.pop("sleep_multiplier", 2)
        while True:
            try:
                return self.generate_one(prompt, sampling_args, **kwargs)
            except Exception as e:
                n_attempts += 1
                error = e
                if n_attempts >= max_attempts:
                    break
                print(f"Failed to generate at attempt {n_attempts} out of {max_attempts}. Sleeping for {sleep_time} seconds.")
                sleep(sleep_time)
                sleep_time *= sleep_multiplier
        print(f"Failed to generate response after {n_attempts} attempts. Raising error.")
        raise error

    def generate_parallel(self, prompts: Union[str, List[str]], sampling_args: 'SamplingArgs' = None, workers: int = 128, **kwargs) -> Union[GenerationLog, List[GenerationLog]]:
        """Generate responses in parallel using a thread pool. Works for list of prompts."""
        if isinstance(prompts, str):
            return self.generate_one_resilient(prompts, sampling_args, **kwargs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.generate_one_resilient, prompt, sampling_args, **kwargs) for prompt in prompts]
            results = [future.result() for future in futures]
        return results

    async def generate_async(self, prompts: Union[str, List[str]], sampling_args: 'SamplingArgs' = None, **kwargs) -> Union[GenerationLog, List[GenerationLog]]:
        """Asynchronously generate responses by running generate_one_resilient in an executor."""
        if isinstance(prompts, str):
            return await asyncio.to_thread(self.generate_one_resilient, prompts, sampling_args, **kwargs)
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self.generate_one_resilient, prompt, sampling_args, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

    def generate(self, prompts: Union[str, List[str]], sampling_args: SamplingArgs = None, **kwargs) -> Union[GenerationLog, List[GenerationLog]]:
        """Generate responses from a given client. Handles both single and multiple prompts."""
        if isinstance(prompts, str):
            return self.generate_one_resilient(prompts, sampling_args, **kwargs)
        else:
            # return asyncio.run(self.generate_async(prompts, sampling_args, **kwargs))
            return self.generate_parallel(prompts, sampling_args, **kwargs)

    def get_monitored_run_func(self, _generate_from_cache: Callable = None, **kwargs) -> Tuple[Callable, List[GenerationLog]]:
        """
            Wrapper that returns a function that records prompts and responses of a calls to the client for generation. 
            This function also supports sampling from the cache (i.e. existing results), for the sake of 
            cost saving. 
        """
        logs = []
        generate_from_cache = _generate_from_cache
        passed_kwargs = kwargs.copy()
        default_sampling_args = passed_kwargs.pop("sampling_args", None)
        
        def run_func(prompts: Union[str, List[str]], use_cache: bool = False, sampling_args: SamplingArgs = None, **kwargs) -> Union[str, List[str]]:
            nonlocal logs, passed_kwargs, generate_from_cache, default_sampling_args

            if sampling_args is None:
                sampling_args = default_sampling_args
            
            if use_cache and generate_from_cache is not None:
                results = generate_from_cache(prompts)
            else:
                results = self.generate(prompts, sampling_args, **passed_kwargs, **kwargs)

            if isinstance(results, GenerationLog):
                logs.append(results)
                return results.response

            logs.extend(results)
            return [result.response for result in results]
        
        return run_func, logs

_client_registry = {}

def register_client(name: str):
    def decorator(cls):
        if name in _client_registry:
            raise ValueError(f"Client '{name}' already registered.")
        _client_registry[name] = cls
        return cls
    return decorator

def get_client(name: str, *args, **kwargs):
    if name not in _client_registry:
        raise ValueError(f"Client '{name}' not found in registry. Available clients: {list(_client_registry.keys())}")
    return _client_registry[name](*args, **kwargs)

def list_clients() -> List[str]:
    """Return a list of all registered client names."""
    return list(_client_registry.keys())
