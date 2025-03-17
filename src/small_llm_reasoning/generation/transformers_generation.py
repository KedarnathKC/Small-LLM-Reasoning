import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

hf_token=os.getenv('hf_token')

def generation(prompts, model_name, padding, padding_side, torch_dtype, special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch_dtype,
        token=hf_token, 
        cache_dir=cache_dir
    )

    if padding_side:
        tokenizer.padding_side = padding_side
    add_special_tokens = {"add_special_tokens": special_tokens}

    # Tokenize with padding for batch input
    inputs = tokenizer(prompts, padding=padding, return_tensors="pt", **add_special_tokens).to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # To avoid running into OOM
    del model
    del tokenizer
    gc.collect()  # Trigger Python's garbage collector
    torch.cuda.empty_cache()  # Free unused GPU memory

    return outputs