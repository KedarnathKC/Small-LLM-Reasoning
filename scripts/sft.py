import os
import re
import json
import torch
import argparse
from peft import LoraConfig
from trl import  SFTConfig, SFTTrainer
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from small_llm_reasoning.trainer.sft_trainer import CustomizedSFTTrainer
from small_llm_reasoning.data.data_sampler import BatchSampler
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM


cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Reading HF Token
hf_token = os.getenv("hf_token")

# We will need formatting_prompt_func as for our M4, M5, M6 methods, if we use standardard sft dataset formats, the trainer will call apply_chat_template which will add an additional
# assistant token that the assistant needs to complete. which wont work for our completion prompts used in M4, M5, M6. 
def formatting_prompts_func_gsm8k(example):
    answer = format_answer_gsm8k(example['answer'])
    text = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {example['question']}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>'
    return text

def format_answer_gsm8k(answer):
        answer = re.sub(r'<<.*?>>', '', answer)
        answer = answer.replace('####', 'The final answer is')
        return answer

def formatting_prompts_func_wnc(example):
    with open('./prompts/neutralization.json') as fp:
        task_prompt = json.load(fp)
    system_msg= f'<|start_header_id|>system<|end_header_id|>\n\n{task_prompt['system_msg']}<|eot_id|>'
    user_msg= f'<|start_header_id|>user<|end_header_id|>\n\n{task_prompt['user_msg'].format(instruction=task_prompt['task_prompt'], question=example['input'])}<|eot_id|>'
    rationale= example['rationale'] if 'rationale' in example else '' # vanilla sft using off-the-shelf data doesn't have rationale
    assistant_msg=f'<|start_header_id|>assistant<|end_header_id|>\n\n{task_prompt['assistant_msg'].format(rationale=rationale, response=example['edits'])}<|eot_id|>'    
    text= system_msg + user_msg + assistant_msg
    return text

def formatting_prompts_func_gec(example):
    with open('./prompts/gec.json') as fp:
        task_prompt = json.load(fp)
    system_msg= f'<|start_header_id|>system<|end_header_id|>\n\n{task_prompt['system_msg']}<|eot_id|>'
    user_msg= f'<|start_header_id|>user<|end_header_id|>\n\n{task_prompt['user_msg'].format(instruction=task_prompt['task_prompt'], question=example['input'])}<|eot_id|>'
    rationale= example['rationale'] if 'rationale' in example else '' # vanilla sft using off-the-shelf data doesn't have rationale
    # For GEC multiple editors are allowed, so when we use off-the-shelf data we need to choose one of the output or else train it as n different examples
    # Using to determine off-the-shelf data or custom data
    if 'rationale' in example:
        assistant_msg=f'<|start_header_id|>assistant<|end_header_id|>\n\n{task_prompt['assistant_msg'].format(rationale=rationale, response=example['output'])}<|eot_id|>' 
    else: # Currently taking first output for off-the-shelf data
        assistant_msg=f'<|start_header_id|>assistant<|end_header_id|>\n\n{task_prompt['assistant_msg'].format(rationale=rationale, response=example['output'][0])}<|eot_id|>' 
    text= system_msg + user_msg + assistant_msg
    return text

formatting_funcs = {
    'gsm8k': formatting_prompts_func_gsm8k,
    'wnc': formatting_prompts_func_wnc,
    'gec': formatting_prompts_func_gec
}

def finetune(model_name, train_data_path, output_dir, formatting_func, add_special_tokens, sample, sampling_ratio, threshold_col, threshold_value, lora, epochs, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    # Loading data
    train_data= load_from_disk(train_data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # TODO:
    # Currently hard-coding the response template. Will require change when we run M4,M5,M6
    # if not response_template:
    #     print("response_template is required. Stopping the program....")
    #     return
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # Deciding which formatting_prompt_func to use
    formatting_prompts_func= formatting_funcs[formatting_func]

    # Set up the trainer
    training_args = SFTConfig(
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "cache_dir":cache_dir
        },
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_ratio=warmup,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=100,
        # Using this 3072(prompt) + 512(output). The 3072(prompt) is taken from LLaMA : https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals?row=0
        max_seq_length  = max_seq_length
    )

    training_args.add_special_tokens = add_special_tokens

    # # Helps in deciding which method to use for getting training batch 
    # batch_sampler=None
    # if sample:
    #     batch_sampler= BatchSampler(
    #         dataset=train_data,
    #         threshold_column=threshold_col,
    #         threshold=threshold_value,
    #         batch_size=training_args.per_device_train_batch_size,
    #         sampling_ratio=sampling_ratio
        # )

    if lora:
        # PEFT config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            # modules_to_save requried : https://huggingface.co/docs/trl/en/sft_trainer#training-adapters
            # Without this, the fine-tuned model will produce unbounded or nonsense generations.
            modules_to_save=["lm_head", "embed_tokens"], 
            task_type="CAUSAL_LM",
        )

        trainer = CustomizedSFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=train_data,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer,
            peft_config=peft_config,
            use_sampling=sample,
            threshold_column=threshold_col,
            threshold=threshold_value,
            sampling_ratio=sampling_ratio
        )
    else:
        trainer = CustomizedSFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=train_data,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer,
            use_sampling=sample,
            threshold_column=threshold_col,
            threshold=threshold_value,
            sampling_ratio=sampling_ratio
        )
         
    # Start training
    trainer.train()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    # Directly hard-coding response-template in the start_finetuning func, due to the issue raised below.
    # parser.add_argument('--response_template', type=str, default=None) 
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--formatting_func", type=str, choices=list(formatting_funcs.keys()), required=True,help='Function to call to format the data during SFT')
    parser.add_argument("--add_special_tokens", action='store_true', help='Set this flag to true', default=True)
    parser.add_argument('--sample', action='store_true', help='Set the flag to true if sampling based data creation is required', default=None)
    parser.add_argument('--sampling_ratio', type=float, default=0.9, help='Sampling amount for below threshold values')
    parser.add_argument('--threshold_col', type=str, default=None, help='Column that needs to be used for thresholding')
    parser.add_argument('--threshold_value', type=float, default=None, help='Threshold value to use for spliting the data based on threshold_col')
    parser.add_argument("--lora", action="store_true", help="Set this flag to true if you want to use LoRA Finetuning", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=500)
    args = parser.parse_args()

    # Need help from Wenlong to verify if this is correct. 99% sure it is correct
    # # Doing this as passing response_template from command line makes the newline character \\n rather than \n. 
    # # Which was causing a issue.
    # if args.response_template:
    #     # Replace literal `\n` with newline character
    #     print(f'Before: {args.response_template}')
    #     args.response_template = args.response_template.encode('utf-8').decode('unicode_escape')
    #     print(f'After: {args.response_template}')
    
    finetune(
        model_name=args.model_name, 
        train_data_path=args.train_data_path, 
        # response_template=args.response_template
        output_dir=args.output_dir, 
        formatting_func=args.formatting_func,
        add_special_tokens=args.add_special_tokens,
        sample=args.sample,
        sampling_ratio=args.sampling_ratio,
        threshold_col=args.threshold_col,
        threshold_value=args.threshold_value,
        lora=args.lora,
        epochs=args.epochs, 
        lr=args.lr, 
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length
    )

if __name__=='__main__':
    main()
