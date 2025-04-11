import os
import re
import torch
import argparse
from datasets import load_from_disk, load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import  SFTConfig, SFTTrainer
from small_llm_reasoning.data.utils import create_data_from_teacher_gen
from small_llm_reasoning.trainer.sft_trainer import CustomizedSFTTrainer
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir


# Loading model
hf_token = os.getenv("hf_token")

# Importing from small_llm_reasoning.data.utils
def formatting_prompts_func(examples):
    # print(f"[DEBUG] formatting input: {type(examples)}")
    # print(f"[DEBUG] example: {examples}")
    # print(f"[DEBUG] type(examples): {type(examples['question'])}")
    answer = format_answer(examples['answer'])
    text = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {examples['question']}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}'
    return text
    # if isinstance(examples, dict) and not isinstance(examples['question'], list):
    #     # Handle single example
    #     answer = format_answer(examples['answer'])
    #     text = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {examples['question']}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}'
    #     return text
    # else:
    #     # Handle batch of examples
    #     formatted_examples = []
    #     for i in range(len(examples['question'])):
    #         answer = format_answer(examples['answer'][i])
    #         text = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {examples['question'][i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}'
    #         formatted_examples.append(text)
    #     return formatted_examples

def format_answer(answer):
        answer = re.sub(r'<<.*?>>', '', answer)
        answer = answer.replace('####', 'The final answer is')
        return answer

def finetune(model_name, train_data_path, teacher_data_path, remove_incorrect, output_dir, lora, epochs, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    '''
    model_name: 
    train_data_path: 
    teacher_data_path:
    remove_incorrect: Remove incorrect teacher predictions from data
    output_dir:
    lora:
    epochs:
    lr:
    lr_scheduler_type:
    warmup:
    weight_decay:
    per_device_train_batch_size:
    gradient_accumulation_steps: 
    max_seq_length: 
    '''

    # Loading data
    data_train= load_from_disk(train_data_path)

    if teacher_data_path:
        data_teacher=load_dataset('json',data_files=teacher_data_path)['train']
        data_train=create_data_from_teacher_gen(data=data_train, teacher_data=data_teacher, remove_incorrects=remove_incorrect)
        print('Teacher Generations replaced as answers')

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
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

    training_args.add_special_tokens = False

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
            train_dataset=data_train,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    else:
        trainer = CustomizedSFTTrainer(
            model=model_name,
            args=training_args,
            train_dataset=data_train,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            tokenizer=tokenizer
        )
         
    # Start training
    trainer.train()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument('--teacher_data_path', type=str, default=None)
    parser.add_argument("--remove_incorrect", action="store_true", help="Set this flag to true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora", action="store_true", help="Set this flag to true", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=500)
    args = parser.parse_args()
    
    finetune(
        model_name=args.model_name, 
        train_data_path=args.train_data_path, 
        teacher_data_path=args.teacher_data_path,
        remove_incorrect=args.remove_incorrect,
        output_dir=args.output_dir, 
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
