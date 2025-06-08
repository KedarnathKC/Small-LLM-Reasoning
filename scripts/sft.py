import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

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

# Reading HF Token
hf_token = os.getenv("hf_token")

# data is in preference-style. So to create sft data just join prompt+chosen
def formatting_func(example):
    input_ids = example['prompt_input_ids'] + example['chosen_input_ids']
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def finetune(model_name, train_data_path, output_dir, sample, sampling_ratio, threshold_col, threshold_value, lora, epochs, max_steps, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    # Loading data
    train_data= load_dataset('json', data_files=train_data_path)['train']
    train_data=train_data.map(lambda ex: formatting_func(ex))

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # TODO:
    # Currently hard-coding the response template. Will require change when we run M4,M5,M6
    # if not response_template:
    #     print("response_template is required. Stopping the program....")
    #     return
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Set up the trainer
    training_args = SFTConfig(
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "cache_dir":cache_dir
        },
        output_dir=output_dir,
        # using max-steps
        # num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_ratio=warmup,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        # For gsm8k: Using this 3072(prompt) + 512(output). The 3072(prompt) is taken from LLaMA : https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals?row=0
        max_seq_length  = max_seq_length
    )

    # training_args.add_special_tokens = add_special_tokens

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
    # We don't require add_special_tokens as we are passing a tokenized data
    # Still keeping it here, when we pass untokenized data. We will then need to pass this to as a TrainingArgument and handle it in CustomSFTTrainer
    # parser.add_argument("--add_special_tokens", action='store_true', help='Set this flag to true', default=True)
    parser.add_argument('--sample', action='store_true', help='Set the flag to true if sampling based data creation is required', default=None)
    parser.add_argument('--sampling_ratio', type=float, default=0.9, help='Sampling amount for below threshold values')
    parser.add_argument('--threshold_col', type=str, default=None, help='Column that needs to be used for thresholding')
    parser.add_argument('--threshold_value', type=float, default=None, help='Threshold value to use for spliting the data based on threshold_col')
    parser.add_argument("--lora", action="store_true", help="Set this flag to true if you want to use LoRA Finetuning", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
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
        sample=args.sample,
        sampling_ratio=args.sampling_ratio,
        threshold_col=args.threshold_col,
        threshold_value=args.threshold_value,
        lora=args.lora,
        epochs=args.epochs, 
        max_steps=args.max_steps,
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
