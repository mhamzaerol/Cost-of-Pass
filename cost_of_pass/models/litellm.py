from litellm import completion
from litellm.utils import get_valid_models
from .base import Client, register_client, SamplingArgs, GenerationLog
import litellm
from pathlib import Path
import json

class LiteLLMClient(Client):
    def generate_one(self, prompt: str, sampling_args: SamplingArgs = None, **kwargs) -> GenerationLog:
        if sampling_args is None:
            sampling_args = SamplingArgs()
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=sampling_args.temperature,
            top_p=sampling_args.top_p,
            max_tokens=sampling_args.max_tokens,
            **kwargs
        )
        model_output = response.choices[0].message.content
        if response.choices[0].message.get("reasoning_content"):
            model_output = f'{response.choices[0].message.reasoning_content}\n{model_output}'
        
        try:
            prompt_n_tokens = response.usage.prompt_tokens
        except AttributeError:
            prompt_n_tokens = self.count_tokens(prompt)

        try:
            response_n_tokens = response.usage.completion_tokens
        except AttributeError:
            response_n_tokens = self.count_tokens(model_output)
        
        return GenerationLog(
            prompt=prompt,
            response=model_output,
            prompt_n_tokens=prompt_n_tokens,
            response_n_tokens=response_n_tokens
        )

    def count_tokens(self, text: str) -> int:
        return litellm.token_counter(
            model=self.model_name, 
            text=text
        )
    
    def get_cost_per_token(self, token_type: str) -> float:
        try:
            costs = litellm.cost_per_token(
                model=self.model_name, 
                prompt_tokens=1,
                completion_tokens=1
            )
            return costs[0] if token_type == "prompt" else costs[1]
        except:
            # In case the cost information is not available in litellm
            with open(Path(__file__).parent / "cost_data" / "litellm.json", "r") as f:
                costs = json.load(f)
            if self.model_name in costs:
                return costs[self.model_name][token_type]
            else:
                print(f"!!!WARNING WARNING WARNING!!! We cannot find the cost for the model: {self.model_name}")
                return -1
        
    @property
    def cost_per_prompt_token(self) -> float:
        return self.get_cost_per_token("prompt")

    @property 
    def cost_per_completion_token(self) -> float:
        return self.get_cost_per_token("completion")


def make_litellm_client(key: str):
    class AutoLiteLLMClient(LiteLLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(model_name=key)
    return AutoLiteLLMClient

for model in get_valid_models():
    key = f"{model}"
    client_cls = make_litellm_client(key)
    register_client(key)(client_cls)
