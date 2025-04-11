import os
import argparse
from datasets import load_from_disk
from sft import finetune

def start_finetuning(model_name, train_data_path, response_template, output_dir, add_special_tokens, lora, epochs, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    print('Loading data...')
    train_data=load_from_disk(train_data_path)
    print('Data loaded')
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
   
    finetune(
        model_name=model_name, 
        train_data=train_data, 
        response_template=response_template, 
        output_dir=output_dir, 
        add_special_tokens=add_special_tokens, 
        lora=lora, 
        epochs=epochs, 
        lr=lr, 
        lr_scheduler_type=lr_scheduler_type, 
        warmup=warmup, 
        weight_decay=weight_decay, 
        per_device_train_batch_size=per_device_train_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps, 
        max_seq_length=max_seq_length
    )

    print(f'Fine-tuning of the model: {model_name} is completed.\nThe checkpoints are saved at: {output_dir}')
    return

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument('--response_template', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--add_special_tokens", action='store_true', help='Set this flag to true', default=False)
    parser.add_argument("--lora", action="store_true", help="Set this flag to true", default=False)
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

    start_finetuning(
        model_name=args.model_name, 
        train_data_path=args.train_data_path, 
        response_template=args.response_template,
        output_dir=args.output_dir, 
        add_special_tokens=args.add_special_tokens,
        lora=args.lora,
        epochs=args.epochs, 
        lr=args.lr, 
        lr_scheduler_type=args.lr_scheduler_type,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length
    )

if __name__=='__main__':
    main()