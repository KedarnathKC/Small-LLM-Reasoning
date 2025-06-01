import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import json
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk

def get_prompt(ex, prompt_template, task_prompt, few_shot, few_shot_examples, input_col, output_col, n=3):
    prompt=[
        {
            'role':'system',
            'content':prompt_template['system_msg']
        }
    ]
    if few_shot:
        for idx in range(n):
            prompt.extend([
                {
                    'role':'user',
                    'content':prompt_template['user_msg'].format(instruction=task_prompt, question=few_shot_examples[idx][input_col])
                },
                {
                    'role':'assistant',
                    'content':prompt_template['assistant_msg'].format(response=few_shot_examples[idx][output_col], rationale=few_shot_examples[idx]['rationale'])
                }
                                                                      
            ])
        
    prompt.append(
        {
            'role':'user',
            'content':prompt_template['user_msg'].format(instruction=task_prompt, question=ex)
        }
    )
    
    return prompt

def tokenize_function(tokenizer, example,input_col, output_col, prompt_template, task_prompt, few_shot, few_shot_examples, n=3):
    prompt= get_prompt(example[input_col], prompt_template, task_prompt, few_shot, few_shot_examples, input_col, output_col, n)
    prompt= tokenizer.apply_chat_template(prompt,  tokenize= False, add_generation_prompt=True)
    return {'input_ids': {'prompt_token_ids':tokenizer(prompt, add_special_tokens=False)['input_ids']}}

def tokenize_data(tokenizer, hf_name, task_name, split, input_col, output_col, prompt_template, task_prompt, few_shot, few_shot_examples, n):
    data_path= f'../datasets/{task_name}'
    data = load_from_disk(f"{data_path}/raw/{split}/")
    # If we pass in batches tokenier adds padding tokens.
    tokenized_dataset = data.map(lambda ex: tokenize_function(tokenizer, ex, input_col, output_col, prompt_template, task_prompt, few_shot, few_shot_examples, n), batched=False)
    output_path=f"{data_path}/tokenized/{hf_name}/{split}/{n}-shot/"
    tokenized_dataset.save_to_disk(output_path)
    print(f'Tokenized data saved at: {output_path}')
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--task', type=str, nargs='+', required=True, help='Task name')
    parser.add_argument('--split', type=str, nargs='+', required=True, help='Split name')
    parser.add_argument('--input_col', type=str, required=True, help='Input column name')
    parser.add_argument('--output_col', type=str, required=True, help='Output column name')
    parser.add_argument("--few_shot", action='store_true', help='Set this flag to true', default=False)
    parser.add_argument("--n", type=int, nargs='+', required=False, help='Number of shots')
    
    args = parser.parse_args() 

    if args.few_shot and args.n is None:
        raise ValueError("Number of shots is required when few-shot is True")
    
    # Loading tokenizer
    hf_token = os.getenv("hf_token")
    hf_name=args.model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    for task in args.task:
        task_prompt_path=f'../prompts/{task}.json'
        with open(task_prompt_path) as fp:
            task_prompt = json.load(fp)
        prompt_template={
            'system_msg':task_prompt['system_msg'],
            'user_msg':task_prompt['user_msg'],
            'assistant_msg':task_prompt['assistant_msg']
        }
        for split in args.split:
            if args.few_shot:
                for n in args.n:
                    tokenize_data(tokenizer, hf_name, task, split, args.input_col, args.output_col, prompt_template, task_prompt['task_prompt'], args.few_shot, task_prompt['few_shot'], n)
            else:
                tokenize_data(tokenizer, hf_name, task, split, args.input_col, args.output_col, prompt_template, task_prompt['task_prompt'], args.few_shot, task_prompt['few_shot'], 0)

if __name__ == "__main__":
    main()