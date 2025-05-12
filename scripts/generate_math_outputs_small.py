import os
import gc
import re
import torch
import argparse
import json
from transformers import AutoTokenizer
from small_llm_reasoning.generation.vllm_generation import llama_forward
from small_llm_reasoning.evaluation.gsm8k import get_score, eight_shot_messages
from datasets import load_from_disk
import pickle


hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("HF_HOME")

def natural_sort_key(checkpoint_path):
    """Extracts numerical part from checkpoint names for correct sorting."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    return int(match.group(1)) if match else float('inf')  # Send non-matching to the end

def generate(model_path, eval_data_path, prompting_strategy, max_tokens, log_probs, temperature, n_samples, n_gpus, output_path, input_key):
    '''
        model_path : 
        eval_data_path : 
        max_length : 
        temperature : 
        n_samples :  
    '''
    # Loading data
    data = load_from_disk(eval_data_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    all_outputs = llama_forward(
        prompts=data[input_key], 
        model_path=model_path, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        n_samples=n_samples,
        n_gpus=n_gpus,
        log_probs=log_probs
    )

    # generated_outputs = [
    #     {
    #         'input': tokenizer.decode(out.prompt_token_ids, skip_special_tokens=False),
    #         'output': [ith_output.text for ith_output in out.outputs],
    #         'token_ids': [list(ith_output.token_ids) for ith_output in out.outputs],
    #         'log_probs': [
    #             [token_log_probs[token_id].logprob for token_log_probs, token_id 
    #             in zip(ith_output.logprobs, ith_output.token_ids)]
    #             for ith_output in out.outputs
    #         ],
    #         'all_returned_log_probs': [
    #             [
    #                 {
    #                     'token_id': token,
    #                     'logprob': logprob.logprob,
    #                     'rank': logprob.rank,
    #                     'decoded_token': logprob.decoded_token
    #                 }
    #                 for token, logprob in token_log_probs.items()
    #             ]
    #             for ith_output in out.outputs
    #             for token_log_probs in ith_output.logprobs
    #         ]
    #     }
    #     for out in all_outputs
    # ]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    all_outputs_path = os.path.join(output_dir, "all_outputs_processed.json")

    all_outputs_processed = [
        {
            'prompt': out.prompt,
            "prompt_token_ids": out.prompt_token_ids,
            'outputs': [ith_output.text for ith_output in out.outputs],
            'token_ids': [list(ith_output.token_ids) for ith_output in out.outputs],
        }
        for out in all_outputs
    ]

    with open(all_outputs_path, "w") as f:
        json.dump(all_outputs_processed, f, indent=4)

    generated_outputs = []
    for out in all_outputs:
        entry = {
            'input': tokenizer.decode(out.prompt_token_ids, skip_special_tokens=False),
            'output': [ith_output.text for ith_output in out.outputs],
            'token_ids': [list(ith_output.token_ids) for ith_output in out.outputs],
        }
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


    generated_outputs_path = os.path.join(output_dir, output_path)
    with open(generated_outputs_path, "w") as f:
        json.dump(generated_outputs, f, indent=4)

    # score = get_score(eval_data_path, generated_outputs_path)
    # print(f"SCORE of {model_path} : ",score)
    # print(f"Output saved in: {generated_outputs_path}")

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
    # No longer require prompting_strategy,as we are using tokenized data directly, which handles 0-shot and 8-shot prompts. 
    parser.add_argument("--prompting_strategy", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--log_probs", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    parser.add_argument("--input_key", type=str, default="four_shot_prompt")
    args = parser.parse_args()       

    if os.path.isdir(args.model_path):
        # Iterate over all checkpoints inside model_path (if local model path)
        checkpoints = [os.path.join(args.model_path, d) for d in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, d))]
        checkpoints = sorted(checkpoints, key=natural_sort_key)
    else:
        # Treat the model_path as a Hugging Face model name
        checkpoints = [args.model_path]

    if not checkpoints:
        print(f"No checkpoints found in {args.model_path}")
        return


    # output_path = f'./outputs/{args.exp_id}/{args.eval_id}/generated_outputs.json'
    eval_id = int(args.eval_id)
    for checkpoint in checkpoints:
        # model_path = os.path.join(args.model_path, os.path.basename(checkpoint))
        model_path = checkpoint
        output_path = f'{args.output_path}/eval_{eval_id}/generated_outputs.json'

        print(f"\nEvaluating model: {model_path}\n")
        generate(
            model_path= model_path,
            prompting_strategy=args.prompting_strategy,
            eval_data_path=args.eval_data_path, 
            max_tokens=args.max_tokens, 
            log_probs=args.log_probs if args.log_probs>0 else None,
            temperature=args.temperature, 
            n_samples=args.n_samples, 
            n_gpus=args.n_gpus,
            output_path=output_path,
            input_key=args.input_key
        )
        eval_id+=1


if __name__=='__main__':
    main()
