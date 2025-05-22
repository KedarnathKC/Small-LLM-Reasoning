import os
import gc
import torch
from typing import Sequence, Optional, Union, Iterable, List, Dict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from small_llm_reasoning.utils import python_utils
# from utils.prompt_utils import generate_all_answer_strings


cache_directory = os.getenv("HF_HOME")
os.environ['HF_HOME'] = cache_directory

def llama_forward(
    prompts: List[List[Dict]]= None,
    model: Optional[str] = None,
    model_path: Optional[str] = None,
    max_tokens: int = 300,  # Generation length
    temperature: float = 0.1,
    n_samples: int = 8,
    n_gpus: int = 4,
    log_probs:int | None = None
) -> Optional[Sequence[str]]:

    print(f"Number of gpus: {n_gpus}")

    assert model is not None or model_path is not None, "model or model_path must be provided"

    if temperature == 0.0:
        # best_of must be 1 when using greedy decoding
        n_samples = 1

    stop_strings =  ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

    sampling_params = SamplingParams(n=n_samples,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop=stop_strings,
                                    #  seed=1,
                                     logprobs=log_probs
                                     )

    # Create an LLM.
    if model is None:
        model = LLM(
            model=model_path, 
            tokenizer=model_path, 
            tensor_parallel_size=n_gpus, 
            download_dir=cache_directory,
            # enable_lora=enable_lora  
        )

    all_outputs = model.generate(
        sampling_params=sampling_params,
        prompts=prompts
    )
    
    # To avoid running into OOM
    del model
    gc.collect()  # Trigger Python's garbage collector
    torch.cuda.empty_cache()  # Free unused GPU memory

    return all_outputs

