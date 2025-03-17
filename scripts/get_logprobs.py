import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token=os.getenv('hf_token')

def combine_data(q,a):
    return q+a
    

def log_probs(model_path, data_path, output_path, padding_strategy, padding_side, batch_size, torch_dtype, special_tokens):
    '''
    model_path: Model name or path to directory containing model weights
    data_path: Path to directory containing HF Dataset
    padding_strategy: Padding strategy for tokenizer
    padding_side: Tokenizer's padding side. If None, default padding side is used
    batch_size: 
    torch_dtype:  
    '''

    # Loading data
    data = load_dataset('json',data_files=data_path)['train']
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if padding_side:
        tokenizer.padding_side = padding_side
    print(f'Special Tokens: {special_tokens}')
    add_special_tokens = {"add_special_tokens": special_tokens}

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch_dtype,
        token=hf_token, 
        cache_dir=cache_dir
    )
    answer_log_probs = []
    for i in tqdm(range(0,data.num_rows,batch_size)):
        questions=data['input'][i:i+batch_size]
        
        prompts=[]
        answers=[]
        for j in range(len(questions)):
            answers.append(data['output'][j][0])
            prompts.append(combine_data(questions[j],answers[j]))
        
        # Tokenize with padding for batch input
        inputs = tokenizer(prompts, padding=padding_strategy, return_tensors="pt", **add_special_tokens).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract logits
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits,dim=-1) # Shape: [batch_size, seq_len, vocab_size]
              
        # Identify the token IDs for the answer (without special tokens)
        answer_token_ids = [tokenizer(a, add_special_tokens=False)["input_ids"] for a in answers]
        answer_tokens = []
        answer_token_idxs=[]
        
        
        for idx, (prompts, answer_tokens_ids) in enumerate(zip(prompts, answer_token_ids)):
            # Extract token IDs for answer tokens in the concatenated input
            answer_start_idx = (inputs["input_ids"][idx] == answer_tokens_ids[0]).nonzero(as_tuple=True)[0][0].item()
            answer_end_idx = answer_start_idx + len(answer_tokens_ids)
            answer_tokens.append(inputs["input_ids"][idx, answer_start_idx:answer_end_idx])
            answer_token_idxs.append(list(range(answer_start_idx, answer_end_idx)))
        
        # Now, extract the log probs for the selected tokens (answer tokens)
        for i, tokens in enumerate(answer_tokens):
            answer_positions = torch.tensor(answer_token_idxs[i], device=model.device)  # Convert to tensor
            selected_log_probs = log_probs[i, answer_positions].gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
            answer_log_probs.append(selected_log_probs.tolist())
    data = data.add_column("log_probs_teacher", answer_log_probs)
    
    # Save as a pretty-printed JSON file
    data.to_json(output_path, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--special_tokens', type=bool, default=False)
    parser.add_argument('--padding_strategy', type=str, default='longest')
    parser.add_argument('--padding_side', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16')
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")

    args = parser.parse_args()   

    output_path= f'./outputs/{args.exp_id}/eval_{args.eval_id}/generated_outputs.json'
    log_probs(
        model_path= args.model_path,
        data_path= args.data_path,
        output_path= output_path,
        padding_strategy= args.padding_strategy,
        padding_side= args.padding_side,
        special_tokens= args.special_tokens,
        batch_size= args.batch_size,
        torch_dtype= args.torch_dtype
    )

if __name__=='__main__':
    main()
