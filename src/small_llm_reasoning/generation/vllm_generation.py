import os
import torch
from typing import Sequence, Optional, Union, Iterable, List, Dict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from small_llm_reasoning.utils import python_utils
# from utils.prompt_utils import generate_all_answer_strings


cache_directory = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache/'
os.environ['HF_HOME'] = cache_directory

def llama_forward(
    prompts: List[List[Dict]],
    model: Optional[str] = None,
    model_path: Optional[str] = None,
    max_tokens: int = 300,  # Generation length
    temperature: float = 0.1,
    n_samples: int = 8,
    n_gpus: int = 8,
    enable_lora: bool = False,
    lora_name: Optional[str] = None,
    lora_int_id: Optional[int] = None,
    lora_path: Optional[str] = None
) -> Optional[Sequence[str]]:
    assert model is not None or model_path is not None, "model or model_path must be provided"

    if temperature == 0.0:
        # best_of must be 1 when using greedy decoding
        n_samples = 1

    stop_strings =  ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

    sampling_params = SamplingParams(n=n_samples,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop=stop_strings)

    # Create an LLM.
    if model is None:
        model = LLM(
            model=model_path, 
            tokenizer=model_path, 
            tensor_parallel_size=n_gpus, 
            enable_lora=enable_lora  
        )
        
    
    # all_outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
    if enable_lora:
        all_outputs = model.chat(
            messages = prompts, 
            sampling_params=sampling_params, 
            use_tqdm=True, 
            lora_request=LoRARequest(
                lora_name=lora_name, 
                lora_int_id=lora_int_id,
                lora_path= lora_path
            )
        )
    else:
        all_outputs = model.chat(messages = prompts, sampling_params=sampling_params, use_tqdm=True)


    return all_outputs