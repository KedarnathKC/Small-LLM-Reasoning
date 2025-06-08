import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import re
import gc
import json
import torch
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from small_llm_reasoning.evaluation import gsm8k, gec, neutralization
from small_llm_reasoning.generation.vllm_generation import llama_forward

hf_token = os.getenv("hf_token")

def natural_sort_key(checkpoint_path):
    """Extracts numerical part from checkpoint names for correct sorting."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    return int(match.group(1)) if match else float('inf')  # Send non-matching to the end

def generate(model_path, eval_data_path, m2_file_path, evaluation_func, input_col, reference_col, max_tokens, log_probs, temperature, top_p, top_k, n_samples, n_gpus, output_path):
    # Loading data
    data = load_from_disk(eval_data_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    all_outputs = llama_forward(
        prompts=data['input_ids'], 
        model_path=model_path, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p,
        top_k=top_k,
        n_samples=n_samples,
        n_gpus=n_gpus,
        log_probs=log_probs
    )

    generated_outputs = []
    for out in all_outputs:
        entry = {
            'prompt': tokenizer.decode(out.prompt_token_ids, skip_special_tokens=False),
            'prompt_input_ids': out.prompt_token_ids,
            'model_output': [ith_output.text for ith_output in out.outputs],
            'model_token_ids': [list(ith_output.token_ids) for ith_output in out.outputs],
        }
        # Not used, calculating using HF scripts.
        if log_probs:
            entry['log_probs'] = [
                [token_log_probs[token_id].logprob for token_log_probs, token_id 
                in zip(ith_output.logprobs, ith_output.token_ids)]
                for ith_output in out.outputs
            ]
            entry['all_returned_log_probs'] = [
                [
                    {
                        'token_id': token,
                        'logprob': logprob.logprob,
                        'rank': logprob.rank,
                        'decoded_token': logprob.decoded_token
                    }
                    for token, logprob in token_log_probs.items()
                ]
                for ith_output in out.outputs
                for token_log_probs in ith_output.logprobs
            ]
        generated_outputs.append(entry)

    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if there is a directory part in the path
        os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    with open(output_path, "w") as f:
        json.dump(generated_outputs, f, indent=4)

    if evaluation_func == 'gsm8k':
        score = gsm8k.get_score(eval_data_path,output_path)  
    elif evaluation_func == 'gec':
        score = gec.get_score(eval_data_path,output_path, m2_file_path, reference_col)  
    elif evaluation_func == 'neutralization':
        score = neutralization.get_score(eval_data_path,output_path, input_col, reference_col)  

    print(f"SCORE of {model_path} : ",score)
    print(f"Output saved in: {output_path}")

    # To avoid running into OOM
    del tokenizer
    del generated_outputs
    
    gc.collect()  # Trigger Python's garbage collector
    torch.cuda.empty_cache()  # Free unused GPU memory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--m2_file_path', type=str)
    parser.add_argument("--evaluation_func", type=str, help='Function to call to evaluate the generate answers.')
    parser.add_argument('--input_col', type=str, help='Name of the input text col that needs to be modified')
    parser.add_argument('--reference_col', type=str, help='Name of the reference text col that needs to used for evaluation')
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--log_probs", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    args = parser.parse_args()      

    if args.evaluation_func == 'gec' and args.m2_file_path is None:
        raise ValueError("m2_file_path is required for GEC evaluation")

    if os.path.isdir(args.model_path):
        # Iterate over all checkpoints inside model_path (if local model path)
        checkpoints = [os.path.join(args.model_path, d) for d in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, d))]
        checkpoints = sorted(checkpoints, key=natural_sort_key)
    else:
        # Treat the model_path as a Hugging Face model name
        checkpoints = [args.model_path]

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {args.model_path}")

    eval_id = int(args.eval_id)
    for checkpoint in checkpoints:
        model_path = checkpoint
        output_path = f'{args.output_path}/eval_{eval_id}/generated_outputs.json'

        print(f"\nEvaluating model: {model_path}\n")
        generate(
            model_path= model_path,
            eval_data_path=args.eval_data_path, 
            m2_file_path=args.m2_file_path,
            evaluation_func=args.evaluation_func,
            input_col=args.input_col,
            reference_col=args.reference_col,
            max_tokens=args.max_tokens, 
            log_probs=args.log_probs if args.log_probs>0 else None,
            temperature=args.temperature, 
            top_p=args.top_p,
            top_k=args.top_k,
            n_samples=args.n_samples, 
            n_gpus=args.n_gpus,
            output_path=output_path
        )
        eval_id+=1


if __name__=='__main__':
    main()
