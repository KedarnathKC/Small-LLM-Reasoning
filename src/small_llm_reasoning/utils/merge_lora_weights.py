import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache/'
os.environ['HF_HOME'] = cache_dir


def add_lora(model_name, lora_path, save_path, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype ,cache_dir=cache_dir)
    
    model = PeftModel.from_pretrained(model, lora_path)

    # Merge LoRA weights into the base model (optional but recommended for inference)
    model = model.merge_and_unload()

    # Save the full model
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("model merged and saved")
    except:
        print("Error while saving the model")

def main():
    model_name= 'meta-llama/Llama-3.2-3B-Instruct'
    checkpoints=[685,1370,2055,2740,3425]
    for checkpoint in checkpoints:
        lora_path= f'./outputs/exp-1.8/checkpoints/checkpoint-{checkpoint}/'
        save_path= f'./outputs/exp-1.8/checkpoints/merged_model-{checkpoint}/'
        torch_dtype='bfloat16'
        add_lora(model_name=model_name, lora_path=lora_path, save_path=save_path, torch_dtype=torch_dtype)

if __name__=='__main__':
    main()
