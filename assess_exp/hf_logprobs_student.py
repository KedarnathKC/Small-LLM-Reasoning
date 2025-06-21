import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import gc
import time
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token=os.getenv('hf_token')

def get_logprobs(model_path, gt_data_path, tokenized_prompt_path, student_generation_path, batch_size, output_path, answer_col, torch_dtype='bfloat16'):
    '''
    model_path:
    gt_data_path:
    tokenized_prompt_path:
    student_generation_path:
    batch_size:
    output_path:
    '''
    # Loading Data
    data_student = load_dataset('json', data_files=student_generation_path)['train']
    data_tokenized = load_from_disk(tokenized_prompt_path)
    data_gt=load_from_disk(gt_data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token, 
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch_dtype,
        token=hf_token, 
        cache_dir=cache_dir
    )
    model.eval()
    
    all_outputs=[]

    for i in tqdm(range(0,data_student.num_rows,batch_size)):
        examples=[]
        questions=[]
        answers=[]
        # make sure we don't go past the end:
        end = min(i + batch_size, data_student.num_rows)
        for j in range(i, end):
            question = torch.tensor(data_tokenized['input_ids'][j]['prompt_token_ids'], dtype=torch.long).unsqueeze(0)
            answer = torch.tensor(data_student['model_token_ids'][j][0], dtype=torch.long).unsqueeze(0)
            examples.append(torch.cat((question, answer), dim=1).squeeze(dim=0))
            questions.append(question)
            answers.append(answer)
            
        # Pad after concatenation
        examples =tokenizer.pad(
            {"input_ids": examples},
            padding=True,  # Pads to longest sequence in batch
            return_tensors="pt"  # Convert to PyTorch tensor
        )['input_ids'].to(model.device)

        # Forward Pass
        with torch.no_grad():  # Ensure no gradients are computed
            outputs = model(examples)

        probs = torch.log_softmax(outputs.logits, dim=-1).detach()
        probs = probs[:, :-1, :]
        examples = examples[:, 1:]
        gen_probs = torch.gather(probs, 2, examples[:, :, None]).squeeze(-1)

        for j in range(examples.shape[0]):
            answer_start_idx = questions[j].shape[1]-1
            answer_end_idx = answer_start_idx + answers[j].shape[1]
            logprobs=[]
            for token, prob in zip(examples[j][answer_start_idx:answer_end_idx], gen_probs[j][answer_start_idx:answer_end_idx]):
                logprobs.append(prob.item())
            all_outputs.append(
                {
                    'prompt':data_student['prompt'][i+j],
                    'prompt_input_ids':data_student['prompt_input_ids'][i+j],
                    'gt_reference':data_gt[answer_col][i+j],
                    'gt_answer':data_student['gt_answer'][i+j],
                    'student_answer':data_student['model_answer'][i+j],
                    'student_score':data_student['score'][i+j],
                    'student_output':data_student['model_output'][i+j][0], 
                    'rejected_input_ids':data_student['model_token_ids'][i+j][0],
                    'student_log_probs_of_student':logprobs
                }
            )

        # Move tensors to CPU and delete them
        examples = examples.cpu()
        outputs.logits = outputs.logits.cpu()
        probs = probs.cpu()
        gen_probs = gen_probs.cpu()
        
        # Clear memory
        del examples, outputs, probs, gen_probs, logprobs, questions, answers
        gc.collect()  # Trigger Python's garbage collector
        torch.cuda.empty_cache()  # Free unused GPU memory
    
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if there is a directory part in the path
        os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=4)
    
    # Clear all remaining memory
    del data_student, data_tokenized, data_gt, tokenizer, model
    del all_outputs
    gc.collect()  # Trigger Python's garbage collector
    torch.cuda.empty_cache()  # Free unused GPU memory
    
    # Force garbage collection again after a short delay
    time.sleep(1)  # Give some time for memory to be freed
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gt_data_path", type=str, default=None)
    parser.add_argument("--tokenized_prompt_path", type=str, default=None)
    parser.add_argument("--student_generation_path", type=str, default=None) 
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--answer_col', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16')
    args = parser.parse_args() 

    get_logprobs(
        model_path=args.model_path,
        gt_data_path=args.gt_data_path,
        tokenized_prompt_path=args.tokenized_prompt_path,
        student_generation_path=args.student_generation_path,
        answer_col=args.answer_col,
        batch_size=args.batch_size,
        output_path=args.output_path
        )

if __name__=='__main__':
    main()