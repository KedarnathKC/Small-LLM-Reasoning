"""
python scripts/evaluation_math.py --model_path meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path /scratch3/workspace/wenlongzhao_umass_edu-metakd/dev_jay/small-reasoning-lmdata/math/raw_data/train_val_test_splits_dataset_feedback_100/ \
    --split feedback --output_path data/math/outputs/Llama-3.2-3B-Instruct-feedback-100 \
    --prompting_strategy 4-shot --max_tokens 5120 --eval_id 0521 --custom_task_prompt="Problem: " --custom_task_prompt_suffix="Answer: "

python scripts/evaluation_math.py --model_path meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path /scratch3/workspace/wenlongzhao_umass_edu-metakd/dev_jay/small-reasoning-lmdata/math/raw_data/train_val_test_splits_dataset_feedback_100/ \
    --split feedback --output_path data/math/outputs/Llama-3.2-3B-Instruct-feedback-100 \
    --prompting_strategy 4-shot --max_tokens 5120 --eval_id 0521 --use_chat_template --remove_system_prompt

"""


import os
import gc
import re
import torch
import argparse
import json
from transformers import AutoTokenizer
from small_llm_reasoning.generation.vllm_generation import llama_forward
from datasets import load_from_disk
from evaluate_math_sympy import evaluate_problems
import json
import sys

TASK_PROMPT = """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: """

FEW_SHOT_EXAMPLES_LIST_MULTI_TURN = [{'role': 'user', 'content': 'Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.'},
                          {'role': 'assistant', 'content': '## Step 1: Consider the expression inside the first square root\nThe expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.\n\n## Step 2: Consider the expression inside the second square root\nThe expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.\n\n## Step 3: Consider the denominator of the expression\nThe denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.\n\n## Step 4: Combine the results from the previous steps\nCombining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.\n\nThe final answer is: $\\boxed{[2,5)}$. I hope it is correct.'},
                          {'role': 'user', 'content': 'If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$'},
                          {'role': 'assistant', 'content': '## Step 1: Understand the properties of determinants\nWe know that the determinant of a product of two matrices is equal to the product of their determinants.\n\n## Step 2: Apply the property to the given matrices\nUsing the property, we can write $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$.\n\n## Step 3: Substitute the given values\nWe are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$. Substituting these values, we get $\\det (\\mathbf{A} \\mathbf{B}) = (2)(12)$.\n\n## Step 4: Calculate the final answer\nEvaluating the product, we find that $\\det (\\mathbf{A} \\mathbf{B}) = 24$.\n\nThe final answer is: $\\boxed{24}$. I hope it is correct.'},
                          {'role': 'user', 'content': 'Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?'},
                          {'role': 'assistant', 'content': '## Step 1: Calculate the total weight Terrell lifts with two 20-pound weights\nTerrell lifts two 20-pound weights 12 times, so the total weight he lifts is $2 \\cdot 12 \\cdot 20 = 480$ pounds.\n\n## Step 2: Calculate the total weight Terrell lifts with two 15-pound weights for n times\nIf Terrell lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.\n\n## Step 3: Equate the total weight lifted with 15-pound weights to 480 pounds and solve for n\nEquating the total weight lifted with 15-pound weights to 480 pounds, we can solve for $n$: $30n = 480$.\n\n## Step 4: Solve for n\nTo solve for $n$, divide both sides by 30: $n = \\frac{480}{30}$.\n\n## Step 5: Calculate the value of n\nThe value of $n$ is $\\frac{480}{30} = 16$.\n\nThe final answer is: $\\boxed{16}$. I hope it is correct.'},
                          {'role': 'user', 'content': 'If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.'},
                          {'role': 'assistant', 'content': '## Step 1: Understand the problem\nWe are given a system of linear equations with two variables, $x$ and $y$, and two constants, $a$ and $b$. Our goal is to find the ratio $\\frac{a}{b}$, assuming $b$ is nonzero.\n\n## Step 2: Multiply the first equation by a suitable factor\nWe multiply the first equation by $-\\frac{3}{2}$ to obtain an equation with the same coefficients for $y$ as the second equation: $$-\\frac{3}{2}(6x-4y)=a\\Rightarrow 6y-9x=-\\frac{3}{2}a.$$\n\n## Step 3: Equate the right-hand sides of the two equations\nSince we know that $6y-9x=b$ from the second equation, we can equate the right-hand sides of the two equations to obtain $$-\\frac{3}{2}a=b.$$\n\n## Step 4: Solve for the ratio $\\frac{a}{b}$\nFinally, we can solve for the ratio $\\frac{a}{b}$ by dividing both sides of the equation by $b$, giving us $$\\frac{a}{b}=-\\frac{2}{3}.$$\n\nThe final answer is: $\\boxed{-\\frac{2}{3}}. I hope it is correct.'}]

hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("HF_HOME")

def natural_sort_key(checkpoint_path):
    """Extracts numerical part from checkpoint names for correct sorting."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    return int(match.group(1)) if match else float('inf')  # Send non-matching to the end

def remove_system_prompt_fn(prompt):
    prompt_split = prompt.split("<|start_header_id|>")
    prompt_suffix = '<|start_header_id|>'.join(prompt_split[2:])
    prompt_prefix = prompt_split[0] + '<|start_header_id|>'
    return prompt_prefix + prompt_suffix


def make_prompt(data, tokenizer, few_shot_examples=False, use_chat_template=False, remove_system_prompt=False, custom_task_prompt=None, custom_task_prompt_suffix=None):
    """
    Creates prompts for evaluation based on configuration options
    
    Args:
        data: The input problem data
        few_shot_examples: Examples to include as few-shot demonstrations
        use_chat_template: Whether to use the chat template formatting
        custom_task_prompt: Custom task prompt to override the default
        custom_task_prompt_suffix: Custom task prompt suffix that will say something like: 'Answer: '
        remove_system_prompt: Whether to remove the system prompt from the prompt
    
    Returns:
        The formatted prompt
    """
        
    # Use custom task prompt if provided, otherwise use default
    final_task_prompt = custom_task_prompt if custom_task_prompt else TASK_PROMPT
    
    # Construct a proper chat prompt if requested
    if use_chat_template:
        # Create a conversation with few-shot examples if available
        messages = []
        
        # Add few-shot examples if available
        if few_shot_examples:
            for few_shot_example in FEW_SHOT_EXAMPLES_LIST_MULTI_TURN:
                if few_shot_example["role"] == "user":
                    messages.append({"role": "user", "content": final_task_prompt + few_shot_example["content"]})
                else:
                    messages.append({"role": "assistant", "content": few_shot_example["content"]})
        
        # Add the current prompt as the final user message
        messages.append({"role": "user", "content": final_task_prompt + data})
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if remove_system_prompt:
            final_prompt = remove_system_prompt_fn(final_prompt)
        return final_prompt
    else:
        # Return the raw prompt
        full_prompt = ''

        if few_shot_examples:
            for few_shot_example in FEW_SHOT_EXAMPLES_LIST_MULTI_TURN:
                if few_shot_example["role"] == "user":
                    full_prompt += final_task_prompt + few_shot_example["content"] + "\n"
                else:
                    full_prompt += custom_task_prompt_suffix + few_shot_example["content"] + "\n\n"

        full_prompt += final_task_prompt + data + "\n" + custom_task_prompt_suffix
        
        return full_prompt

def generate(model_path, eval_data_path, split, prompting_strategy, max_tokens, log_probs, temperature, n_samples, n_gpus, output_path, use_chat_template=False, remove_system_prompt=False, custom_task_prompt=None, custom_task_prompt_suffix=None):
    '''
        model_path : Path to the model checkpoint
        eval_data_path : Path to the evaluation data
        split : Split to evaluate on (e.g., test, train)
        prompting_strategy : Strategy for prompting (e.g., zero-shot, few-shot)
        max_tokens : Maximum tokens to generate
        log_probs : Whether to return log probabilities
        temperature : Temperature for generation
        n_samples : Number of samples to generate
        n_gpus : Number of GPUs to use
        output_path : Path to save the outputs
        use_chat_template : Whether to use chat template formatting
        remove_system_prompt : Whether to remove the system prompt from the prompt
        custom_task_prompt : Custom task prompt to override default
        custom_task_prompt_suffix : Custom task prompt suffix that will say something like: 'Answer: '
    '''
    # Loading data
    data = load_from_disk(eval_data_path)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Format prompts based on configuration
    formatted_prompts = []
    for item in data[split]['question']:
        prompt = make_prompt(item,
                            tokenizer=tokenizer,
                            use_chat_template=use_chat_template, 
                            custom_task_prompt=custom_task_prompt,
                            custom_task_prompt_suffix=custom_task_prompt_suffix,
                            remove_system_prompt=remove_system_prompt,
                            few_shot_examples=True if prompting_strategy == "4-shot" else False)
        
        formatted_prompts.append(prompt)

    print("Printing the first prompt:")
    print(formatted_prompts[0])


    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    all_outputs_path = os.path.join(output_dir, "all_outputs_processed.json")
    generated_outputs = None
    


    all_outputs = llama_forward(
        prompts=formatted_prompts, 
        model_path=model_path, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        n_samples=n_samples,
        n_gpus=n_gpus,
        log_probs=log_probs
    )
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
    print(f"All outputs saved in: {all_outputs_path}")

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


    generated_outputs_path = os.path.join(output_dir, 'generated_outputs.json')
    with open(generated_outputs_path, "w") as f:
        json.dump(generated_outputs, f, indent=4)

    evaluation_results = evaluate_problems(output_dir, eval_data_path, split)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print("Overall Performance:")
    print(f"Exact match rate: {evaluation_results['overall']['exact_match']:.2%}")
    print(f"Total problems: {evaluation_results['overall']['total_problems']}")
    print(f"Correct answers: {evaluation_results['overall']['correct_count']}")

    os.makedirs(output_dir, exist_ok=True)
    # Save results to evals.json in the same directory
    evals_file = os.path.join(output_dir, "evals.json")
    try:
        with open(evals_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)
    except Exception as e:
        print(f"Error writing evals.json: {e}")
        sys.exit(1)

    # score = get_score(eval_data_path, generated_outputs_path)
    # print(f"SCORE of {model_path} : ",score)
    # print(f"Output saved in: {generated_outputs_path}")

    # To avoid running into OOM
    del tokenizer
    del generated_outputs
    
    gc.collect()  # Trigger Python's garbage collector
    torch.cuda.empty_cache()  # Free unused GPU memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="test")
    parser.add_argument('--output_path', type=str, required=True)
    # No longer require prompting_strategy,as we are using tokenized data directly, which handles 0-shot and 8-shot prompts. 
    parser.add_argument("--prompting_strategy", choices=["0-shot", "4-shot"], default="0-shot", required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--log_probs", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    # New arguments for configurable prompt formatting
    parser.add_argument("--use_chat_template", action="store_true", help="Whether to use the model's chat template")
    parser.add_argument("--remove_system_prompt", action="store_true", help="Whether to remove the system prompt from the prompt. This will only apply when use_chat_template is True.")
    parser.add_argument("--custom_task_prompt", type=str, default=None, help="Custom task prompt to override the default")
    parser.add_argument("--custom_task_prompt_suffix", type=str, default=None, help="Custom task prompt suffix that will say something like: 'Answer: '")

    args = parser.parse_args()

    if args.custom_task_prompt:
        assert args.custom_task_prompt_suffix is not None, "custom_task_prompt_suffix must be provided if custom_task_prompt is provided"

    if os.path.isdir(args.model_path):
        # Iterate over all checkpoints inside model_path (if local model path)
        checkpoints = [os.path.join(args.model_path, d) for d in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, d))]
        checkpoints = sorted(checkpoints, key=natural_sort_key)
    else:
        # Treat the model_path as a Hugging Face model name
        checkpoints = [args.model_path]

    if not checkpoints:
        print(f"No checkpoints found in {args.model_path}")
        exit(1)

    # output_path = f'./outputs/{args.exp_id}/{args.eval_id}/generated_outputs.json'
    eval_id = int(args.eval_id)
    for checkpoint in checkpoints:
        # model_path = os.path.join(args.model_path, os.path.basename(checkpoint))
        model_path = checkpoint
        output_path = f'{args.output_path}/eval_{eval_id}/'
        os.makedirs(output_path, exist_ok=True)

        print(f"\nEvaluating model: {model_path}\n")
        generate(
            model_path=model_path,
            prompting_strategy=args.prompting_strategy,
            eval_data_path=args.eval_data_path, 
            split=args.split,
            max_tokens=args.max_tokens, 
            log_probs=args.log_probs if args.log_probs>0 else None,
            temperature=args.temperature, 
            n_samples=args.n_samples, 
            n_gpus=args.n_gpus,
            output_path=output_path,
            use_chat_template=args.use_chat_template,
            remove_system_prompt=args.remove_system_prompt,
            custom_task_prompt=args.custom_task_prompt,
            custom_task_prompt_suffix=args.custom_task_prompt_suffix
        )
        eval_id+=1
