import os
import json
from evaluate import load
from datetime import datetime

import openai
import os
from typing import Dict, List, Optional, Union
import logging
import requests
from tqdm import tqdm
import hashlib
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
logger = logging.getLogger(__name__)

def get_problem_hash(problem): 
    return hashlib.sha256(problem.encode()).hexdigest()

class API:
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the API client with the specified provider.
        
        Args:
            provider (str): The provider to use ('openai', 'together', 'ollama', 'llama')
            api_key (str, optional): API key for the provider. If not provided, will look for environment variable.
        """
        self.provider = provider.lower()
        assert self.provider in ["openai", "llama", "ollama", "together"]
        
        # Set up provider-specific API keys
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        elif self.provider == "together":
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("No Together API key provided. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
            self.base_url = "https://api.together.xyz"
            
        elif self.provider == "llama":
            self.api_key = api_key or os.getenv("LLAMA_API_KEY")
            if not self.api_key:
                raise ValueError("No Llama API key provided. Set LLAMA_API_KEY environment variable or pass api_key parameter.")
            self.base_url = "https://api.llama.com/compat"
            
        elif self.provider == "ollama":
            # Ollama doesn't require an API key as it runs locally
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.api_key = "ollama"

        self.client = openai.OpenAI(base_url=self.base_url + "/v1" if self.base_url else None , api_key=self.api_key)
            
    def get_completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        logprobs: bool = False,
        **kwargs
    ) -> Dict:
        """
        Get a text completion from the specified model.
        
        Args:
            prompt (str): The input prompt
            model (str): The model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            logprobs (bool): Whether to return log probabilities
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict: Response containing completion and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                **kwargs
            )
            return {
                "text": response.choices[0].message.content,
                "logprobs": response.choices[0].logprobs if logprobs else None,
                "model": model,
                "provider": self.provider
            }
                
        except Exception as e:
            logger.error(f"Error getting completion from {self.provider}: {str(e)}")
            raise
        
    def list_models(self) -> List[str]:
        """
        List available models for the current provider.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            if self.provider == "openai" or self.provider == "llama":
                models = self.client.models.list()
                return [model.id for model in models.data]
            
            elif self.provider == "together":

                headers = {
                    "accept": "application/json",
                    "authorization": f"Bearer {self.api_key}"
                }

                response = requests.get(self.base_url + "/models", headers=headers)
                response.raise_for_status()
                response = response.json()
                return [model["id"] for model in response]
                
            elif self.provider == "ollama":
                response = requests.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return [model["name"] for model in response.json()["models"]]

            else:
                logger.warning(f"Unsupported provider: {self.provider}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing models for {self.provider}: {str(e)}")
            raise

provider = "llama"
model_name = "Llama-3.3-70B-Instruct"
# Time string like 20250510-1230
time_str = datetime.now().strftime("%Y%m%d-%H%M")


api = API(provider="llama")

path_to_raw_data = "data/math/raw_data"
output_base_path = "data/math/outputs"
final_output_path = os.path.join(output_base_path, f"{provider}-{model_name}-{time_str}")

split = "test"

path_to_data = os.path.join(path_to_raw_data, split)

final_data = {}

# Load all the pairs in the data
for folder in tqdm(os.listdir(path_to_data), desc="Loading data", total=len(os.listdir(path_to_data))):
    if not os.path.isdir(os.path.join(path_to_data, folder)):
        continue
    final_data[folder] = []
    for file in tqdm(os.listdir(os.path.join(path_to_data, folder)), desc=f"Loading data for {folder}", total=len(os.listdir(os.path.join(path_to_data, folder)))):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(path_to_data, folder, file), "r") as f:
            data = json.load(f)

        final_data[folder].append(data)

test = False

if test:
    final_output_path = f"{final_output_path}-test"
    for key in final_data.keys():
        final_data[key] = final_data[key][:2]
        
rerun_dirs = ['llama-Llama-3.3-70B-Instruct-20250511-0443',
                'llama-Llama-3.3-70B-Instruct-20250512-0000',
                'llama-Llama-3.3-70B-Instruct-20250511-0826',
                'llama-Llama-3.3-70B-Instruct-20250512-0711']
final_rerun_dirs = []
for x in rerun_dirs:
    final_rerun_dirs.append(f'data/math/outputs/{x}')

visited_problems = set()
for rerun_dir in final_rerun_dirs:
    if len(rerun_dir) > 0 and os.path.exists(rerun_dir):
        # Check if results.json exists or results.json does
        old_results_file = os.path.join(rerun_dir, "results.json")
        if os.path.exists(os.path.join(rerun_dir, "results.json")):
            pass
        elif os.path.exists(os.path.join(rerun_dir, "results_partial.json")):
            old_results_file = os.path.join(rerun_dir, "results_partial.json")
        else:
            continue
        
        old_results = json.load(open(old_results_file))
        for key in old_results.keys():
            for result in old_results[key]:
                if result["response"] is None or len(result["response"]) == 0:
                    continue
                visited_problems.add(get_problem_hash(result["problem"]))

print(f"Found {len(visited_problems)} problems in {len(final_rerun_dirs)} rerun directories")


prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: "

top_p = 0
seed = 42
max_gen_len = 5120
max_prompt_len = 3072
num_few_shot = 0
num_generations = 1
prompt_fn = "functools.partial(<function jinja_dialog_format at 0x7f1a897928c0>, template={'prompt': 'Solve the following math problem efficiently and clearly:\\n\\n- For simple problems (2 steps or fewer):\\nProvide a concise solution with minimal explanation.\\n\\n- For complex problems (3 steps or more):\\nUse this step-by-step format:\\n\\n## Step 1: [Concise description]\\n[Brief explanation and calculations]\\n\\n## Step 2: [Concise description]\\n[Brief explanation and calculations]\\n\\n...\\n\\nRegardless of the approach, always conclude with:\\n\\nTherefore, the final answer is: $\\\\boxed{answer}$. I hope it is correct.\\n\\nWhere [answer] is just the final number or expression that solves the problem.\\n\\nProblem: {{ problem }}\\n', 'answer': '{{ solution }}. I hope it is correct.'}, append_gen_prefix=False)"
return_logprobs = False
temperature = 0.0
top_k = 0

results = {}
failed_examples = {}
step = 0

os.makedirs(final_output_path, exist_ok=True)

for key in tqdm(final_data.keys(), desc="Processing data", total=len(final_data.keys())):
    results[key] = []
    failed_examples[key] = []
    for data in tqdm(final_data[key], desc=f"Processing data for {key}", total=len(final_data[key])):
        if get_problem_hash(data["problem"]) in visited_problems:
            continue
        result = {
            "problem": data["problem"],
            "solution": data["solution"],
            "type": data["type"],
            "level": data["level"],
            "prompt": prompt,
        }
        try:
            response = api.get_completion(prompt + data["problem"], model=model_name, max_tokens=max_gen_len, temperature=temperature, top_p=top_p, seed=seed)
            response = response["text"]
            result["response"] = response
            results[key].append(result)
        except Exception as e:
            print(e)
            failed_examples[key].append(result)
        finally:
            step += 1

            if step % 5 == 0:
                with open(os.path.join(final_output_path, f"results_partial.json"), "w") as f:
                    json.dump(results, f, indent=4)

                with open(os.path.join(final_output_path, f"failed_examples_partial.json"), "w") as f:
                    json.dump(failed_examples, f, indent=4)

                print(f"Saved results and failed examples for {step} examples at {final_output_path}")

    with open(os.path.join(final_output_path, f"results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(final_output_path, f"failed_examples.json"), "w") as f:
        json.dump(failed_examples, f, indent=4)

with open(os.path.join(final_output_path, f"results.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(os.path.join(final_output_path, f"failed_examples.json"), "w") as f:
    json.dump(failed_examples, f, indent=4)














