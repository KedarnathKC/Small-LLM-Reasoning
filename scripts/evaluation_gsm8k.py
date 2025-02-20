import os
import argparse
import json
from small_llm_reasoning.generation.vllm_generation import llama_forward
from small_llm_reasoning.evaluation.gsm8k import get_score, eight_shot_messages
from datasets import load_from_disk
from tqdm import tqdm


def generate(model_path, eval_data_path, prompting_strategy, enable_lora, lora_name, lora_int_id, lora_path, max_tokens, temperature, n_samples, n_gpus, output_path):
    '''
        model_path : 
        eval_data_path : 
        max_length : 
        temperature : 
        n_samples :  
    '''
    # Loading data
    data = load_from_disk(eval_data_path)

    
    prompts = []
    for i in tqdm(range(0, len(data["question"])), desc="Processing questions"):
        prompt = [
            {
                'role': 'user',
                'content': f'Given the following problem, reason and give a final answer to the problem.\nProblem: {data["question"][i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
            }
        ]
        prompt = eight_shot_messages + prompt if prompting_strategy == '8-shot' else prompt
        prompts.append(prompt)
        
        
    all_outputs = llama_forward(prompts=prompts, model_path=model_path, max_tokens=max_tokens, temperature=temperature, n_samples=n_samples, n_gpus=1)
    
    generated_outputs=[]
    for ex_outputs in all_outputs:
        generated_outputs.append({
            "input": ex_outputs.prompt, 
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--prompting_strategy", type=str, default=None)
    parser.add_argument("--lora", action="store_true", help="Set this flag to true", default=None)
    parser.add_argument("--lora_name", type=str, help="Human identifiable name", default=None)
    parser.add_argument("--lora_int_id", type=int, help="Global unique ID", default=None)
    parser.add_argument("--lora_path", type=str, help="Path to the LoRA adapter")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    args = parser.parse_args()
    
    output_path = f'./outputs/{args.exp_id}/{args.eval_id}/generated_outputs.json'

    generate(
        model_path=args.model_path, 
        prompting_strategy=args.prompting_strategy,
        eval_data_path=args.eval_data_path, 
        enable_lora=args.lora,
        lora_name=args.lora_name,
        lora_int_id=args.lora_int_id,
        lora_path=args.lora_path,
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        n_samples=args.n_samples, 
        n_gpus=args.n_gpus,
        output_path=output_path
    )


if __name__=='__main__':
    main()
