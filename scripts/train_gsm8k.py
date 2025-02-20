import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from trl.trainer import DataCollatorForCompletionOnlyLM
import re
from peft import LoraConfig
from small_llm_reasoning.data.utils import fromatting_prompts_func, format_answer

# Set up the transformers_cache
cache_dir = '../transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def train(model_name, train_data_path, hf_token, output_path, batch_size, lora=False):
    '''
        model_name : Specifies the name of the model to finetune
        train_data_path : Specifies the location of training data
        hf_token : Hugging Face token 
        output_path : Specifies the output directory
        batch_size : Batch size
        lora : Specifies whether to do LoRA finetuning or full model finetuning
    '''

    # Loading data
    data_train = load_from_disk(train_data_path)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if 'llama' in config.model_type:
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise NotImplementedError
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Set up the trainer
    training_args = SFTConfig(
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "cache_dir": cache_dir
        },
        output_dir="./outputs/LLaMA1B/train_exp_2_Full",
        num_train_epochs=3,
        learning_rate=2e-5,
        # batch-size = 16 -> 1B, 8 -> 3B
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        logging_steps=100,
        # Using this 3072(prompt) + 512(output). The 3072(prompt) is taken from LLaMA : https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals?row=0
        max_seq_length  = 500 
    )

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

        trainer = SFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=data_train,
            formatting_func=fromatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )


    else:
        trainer = SFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=data_train,
            formatting_func=fromatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer
        )
         
    # Start training
    trainer.train()




def main():
    # Loading model
    hf_token = os.getenv("hf_token")

    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--data_path")
    parser.add_argument("--lora", action="store_true", help="Set this flag to true")
    parser.add_argument("--exp_id")
    parser.add_argument("--batch_size")

    
    # Read arguments from command line
    args = parser.parse_args()

    model_name = args.model
    train_data_path = args.data_path
    lora = args.lora
    output_path = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/Small-LLM-Reasoning/outputs/'+args.exp_id
    batch_size = args.batch_size

    train(model_name=model_name, train_data_path=train_data_path, hf_token=hf_token, output_path=output_path, batch_size=batch_size, lora=lora)

if __name__=='__main__':
    main()