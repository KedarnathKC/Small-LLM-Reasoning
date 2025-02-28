import os
import argparse
import json
from transformers import AutoTokenizer
from small_llm_reasoning.generation.vllm_generation import llama_forward
from small_llm_reasoning.evaluation.gsm8k import get_score, eight_shot_messages
from datasets import load_from_disk
from tqdm import tqdm


cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

hf_token = os.getenv("hf_token")

def generate(model_path, eval_data_path, prompting_strategy, max_tokens, temperature, n_samples, n_gpus, output_path):
    '''
        model_path : 
        eval_data_path : 
        max_length : 
        temperature : 
        n_samples :  
    '''
    # Loading data
    data = load_from_disk(eval_data_path)

    
    # prompts = []
    # for i in tqdm(range(0, len(data["question"])), desc="Processing questions"):
    #     prompt = [
    #         {
    #             'role': 'user',
    #             'content': f'Given the following problem, reason and give a final answer to the problem.\nProblem: {data["question"][i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    #         }
    #     ]
    #     prompt = eight_shot_messages + prompt if prompting_strategy == '8-shot' else prompt
    #     prompts.append(prompt)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    all_outputs = llama_forward(
        prompts=data['input_ids'], 
        model_path=model_path, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        n_samples=n_samples,
        n_gpus=1,
    )
    
    generated_outputs=[]
    for ex_outputs in all_outputs:
        generated_outputs.append({
            "input": tokenizer.decode(ex_outputs.prompt_token_ids, skip_special_tokens=False), 
            "output": [
                ith_output.text for ith_output in ex_outputs.outputs
            ]    
        })
    
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if there is a directory part in the path
        os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    with open(output_path, "w") as f:
        json.dump(generated_outputs, f, indent=4)

    score = get_score(eval_data_path,output_path)
    print(f"SCORE of {model_path} : ",score)

    # To avoid running into OOM
    del tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--prompting_strategy", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    args = parser.parse_args()


    # Iterate over all checkpoints inside model_path
    checkpoints = sorted([os.path.join(args.model_path, d) for d in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, d))])

    if not checkpoints:
        print(f"No checkpoints found in {args.model_path}")
        
    # output_path = f'./outputs/{args.exp_id}/{args.eval_id}/generated_outputs.json'
    eval_id = int(args.eval_id)
    for checkpoint in checkpoints:
        model_path = os.path.join(args.model_path, os.path.basename(checkpoint))
        output_path = f'./outputs/{args.exp_id}/eval_{eval_id}/generated_outputs.json'

        print(f"\nEvaluating model: {model_path}\n")
        generate(
            model_path= model_path,
            prompting_strategy=args.prompting_strategy,
            eval_data_path=args.eval_data_path, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            n_samples=args.n_samples, 
            n_gpus=args.n_gpus,
            output_path=output_path
        )
        eval_id+=2


if __name__=='__main__':
    main()
