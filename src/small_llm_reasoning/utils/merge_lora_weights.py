import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache/'
os.environ['HF_HOME'] = cache_dir


def add_lora(model_name, lora_path, save_path, torch_dtype):
    print(f"Merging model at {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype ,cache_dir=cache_dir)
    
    model = PeftModel.from_pretrained(model, lora_path)

    # Merge LoRA weights into the base model (optional but recommended for inference)
    model = model.merge_and_unload()

    # Save the full model
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"model merged and saved at {save_path}")
    except:
        print("Error while saving the model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--checkpoints_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')
    args = parser.parse_args()

    checkpoints = sorted([os.path.join(args.checkpoints_path, d) for d in os.listdir(args.checkpoints_path) if os.path.isdir(os.path.join(args.checkpoints_path, d))])
    
    if not checkpoints:
        print(f"No checkpoints found in {args.model_path}")
        
    for checkpoint in checkpoints:
        add_lora(
            model_name= args.base_model_name,
            lora_path= os.path.join(args.checkpoints_path, os.path.basename(checkpoint)),
            save_path= os.path.join(args.save_path, os.path.basename(checkpoint)),
            torch_dtype= args.torch_dtype
        )



if __name__=='__main__':
    main()
