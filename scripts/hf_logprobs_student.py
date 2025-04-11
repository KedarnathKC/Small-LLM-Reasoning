import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import json
import argparse
import gc
import torch
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token=os.getenv('hf_token')

def get_logprobs(model_path, gt_data_path, tokenized_data_path, student_data_path, batch_size, output_path, torch_dtype='bfloat16'):
    '''
    model_path:
    gt_data_path:
    tokenized_data_path:
    student_data_path:
    batch_size:
    output_path:
    '''
    # Loading Data
    data_student = load_dataset('json', data_files=student_data_path)['train']
    data_tokenized = load_from_disk(tokenized_data_path)
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
        for j in range(i,i+batch_size):
            question = torch.tensor(data_tokenized['input_ids'][j]['prompt_token_ids'], dtype=torch.long).unsqueeze(0)
            answer = torch.tensor(data_student['token_ids'][j][0], dtype=torch.long).unsqueeze(0)
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
                    'prompt':data_student['input'][i+j],
                    'gt_reasoning':data_gt['answer'][i+j],
                    'gt_answer':data_student['GT_Answer'][i+j],
                    'student_token_ids':data_student['token_ids'][i+j][0],
                    'student_reasoning':data_student['output'][i+j][0],
                    'student_answer':data_student['model_answer'][i+j],
                    'student_correctness':data_student['score'][i+j],
                    'student_log_probs':logprobs
                }
            )

        # Clearing memory to avoid OOM issues
        del examples, outputs, probs, gen_probs, logprobs, questions, answers
        gc.collect()  # Trigger Python's garbage collector
        torch.cuda.empty_cache()  # Free unused GPU memory
    
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if there is a directory part in the path
        os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gt_data_path", type=str, default=None)
    parser.add_argument("--tokenized_data_path", type=str, default=None)
    parser.add_argument("--student_data_path", type=str, default=None) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16')
    parser.add_argument("--exp_id", type=str, help="Used in the output path, e.g., exp-1.1")
    parser.add_argument("--eval_id", type=str, help="Used in the output path, e.g., eval-1")
    args = parser.parse_args() 

    output_path = f'./outputs/{args.exp_id}/eval_{args.eval_id}/logprobs.json'

    get_logprobs(
        model_path=args.model_path,
        gt_data_path=args.gt_data_path,
        tokenized_data_path=args.tokenized_data_path,
        student_data_path=args.student_data_path,
        batch_size=args.batch_size,
        output_path=output_path
        )

if __name__=='__main__':
    main()