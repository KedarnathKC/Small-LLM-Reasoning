import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'
hf_token=os.getenv('hf_token')

import argparse
from datasets import load_from_disk, Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from small_llm_reasoning.trainer.dpo_trainer import CustomDPOTrainer


def train( model_name, train_data_path, output_dir, torch_dtype, sample, sampling_ratio, threshold_col, threshold_value, epochs, max_steps, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_length):
    '''
    '''

    # Load data
    train_data= load_dataset('json', data_files=train_data_path)['train']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.add_special_tokens=add_special_tokens

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch_dtype,
        token=hf_token, 
        cache_dir=cache_dir
    )
    # Training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr, # Default 1e-6
        lr_scheduler_type=lr_scheduler_type,        
        weight_decay=weight_decay,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=warmup,
        logging_steps=10,
        dataloader_drop_last=False,
        max_length=max_length
    )

    # Initialize trainer
    trainer = CustomDPOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_data,
        use_sampling=sample,
        threshold_column=threshold_col,
        threshold=threshold_value,
        sampling_ratio=sampling_ratio
        )

    # Train
    trainer.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16')
    # parser.add_argument("--add_special_tokens", action='store_true', help='Set this flag to true', default=False)
    parser.add_argument('--sample', action='store_true', help='Set the flag to true if sampling based data creation is required', default=None)
    parser.add_argument('--sampling_ratio', type=float, default=0.9, help='Sampling amount for below threshold values')
    parser.add_argument('--threshold_col', type=str, default=None, help='Column that needs to be used for thresholding')
    parser.add_argument('--threshold_value', type=float, default=None, help='Threshold value to use for spliting the data based on threshold_col')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=500)
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        # add_special_tokens=args.add_special_tokens,
        sample=args.sample,
        sampling_ratio=args.sampling_ratio,
        threshold_col=args.threshold_col,
        threshold_value=args.threshold_value,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length
    )

if __name__=='__main__':
    main()
