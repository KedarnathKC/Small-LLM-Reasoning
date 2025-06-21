import os
cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/transformers_cache'
os.environ['HF_HOME']=cache_dir
os.environ['HF_HUB_CACHE']=cache_dir+'/hub'

import numpy as np
import gc
import time
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token=os.getenv('hf_token')

def calculate_token_level_logprobs_ratio(teacher_logprobs, student_logprobs):
    teacher_logprobs = np.array(teacher_logprobs)
    student_logprobs = np.array(student_logprobs)
    return np.subtract(teacher_logprobs, student_logprobs).tolist()
    
    # token_level_logprobs=[]
    # for i in range(len(student_logprobs)):
    #     tr_logprobs= np.array(teacher_logprobs[i])
    #     stu_logprobs= np.array(student_logprobs[i])       
    #     token_level_logprobs.append(np.subtract(tr_logprobs,stu_logprobs))
    # return token_level_logprobs

def calculate_sentence_level_logprobs_ratio(teacher_logprobs, student_logprobs):
    teacher_logprob = np.mean(teacher_logprobs)
    student_logprob = np.mean(student_logprobs)
    return np.subtract(teacher_logprob, student_logprob)
    # sentence_level_logprobs=[]
    # teacher_sentence_logprobs=[]
    # student_sentence_logprobs=[]
    # for i in range(len(student_logprobs)):
    #     teacher_sentence_logprobs.append(np.mean(np.array(teacher_logprobs[i])))
    #     student_sentence_logprobs.append(np.mean(np.array(student_logprobs[i])))
    # sentence_level_logprobs=  np.subtract(np.array(teacher_sentence_logprobs),np.array(student_sentence_logprobs))
    # return sentence_level_logprobs

def get_logprobs(model_path, tokenized_prompt_path, student_generation_path, teacher_generation_path, student_logprobs_path, batch_size, output_path, torch_dtype='bfloat16'):
    '''
    model_path:
    tokenized_prompt_path:
    student_generation_path:
    teacher_generation_path:
    batch_size:
    output_path:
    '''
    # Loading Data
    data_student = load_dataset('json', data_files=student_generation_path)['train']
    data_tokenized = load_from_disk(tokenized_prompt_path)
    data_teacher = load_dataset('json', data_files=teacher_generation_path)['train']    
    student_logprobs = load_dataset('json', data_files=student_logprobs_path)['train'] 

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

    all_outputs = []
    # teacher_log_probs_of_student=[]
    # student_log_probs_of_student=[]
    for i in tqdm(range(0,data_student.num_rows,batch_size)):
        examples=[]
        questions=[]
        answers=[]
        # make sure we don’t go past the end:
        end = min(i + batch_size, data_student.num_rows)
        for j in range(i, end):
            # We are using token_ids since all models are from the same family, they will have similar tokenizer's. If that is not the case then you need to 
            # concatinate the models prompt and output using untokenized text.   
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
            teacher_logprobs=[]
            for token, prob in zip(examples[j][answer_start_idx:answer_end_idx], gen_probs[j][answer_start_idx:answer_end_idx]):
                teacher_logprobs.append(prob.item())
            all_outputs.append(
                {
                    'prompt':student_logprobs['prompt'][i+j],
                    'prompt_input_ids':student_logprobs['prompt_input_ids'][i+j],
                    'gt_reference':student_logprobs['gt_reference'][i+j],
                    # evaluation
                    'gt_answer':student_logprobs['gt_answer'][i+j],
                    'student_answer':student_logprobs['student_answer'][i+j],
                    'student_score':student_logprobs['student_score'][i+j],
                    'teacher_answer':data_teacher['model_answer'][i+j],
                    'teacher_score':data_teacher['score'][i+j],
                    # assessment
                    'teacher_output':data_teacher['model_output'][i+j][0],
                    'student_output':student_logprobs['student_output'][i+j], 
                    'chosen_input_ids':data_teacher['model_token_ids'][i+j][0],
                    'rejected_input_ids':student_logprobs['rejected_input_ids'][i+j],
                    'student_log_probs_of_student':student_logprobs['student_log_probs_of_student'][i+j],
                    'teacher_log_probs_of_student':teacher_logprobs,
                    'teacher_student_token_log_prob_ratio':calculate_token_level_logprobs_ratio(teacher_logprobs, student_logprobs['student_log_probs_of_student'][i+j]),
                    'teacher_student_sent_log_prob_ratio':calculate_sentence_level_logprobs_ratio(teacher_logprobs, student_logprobs['student_log_probs_of_student'][i+j])
                }
            )
            # teacher_log_probs_of_student.append(teacher_logprobs)
            # student_log_probs_of_student.append(student_logprobs['student_log_probs_of_student'][i+j])

        # Move tensors to CPU and delete them
        examples = examples.cpu()
        outputs.logits = outputs.logits.cpu()
        probs = probs.cpu()
        gen_probs = gen_probs.cpu()
        
        # Clear memory
        del examples, outputs, probs, gen_probs, teacher_logprobs, questions, answers
        gc.collect()  # Trigger Python's garbage collector
        torch.cuda.empty_cache()  # Free unused GPU memory

    # Calculate the token and sentence level logprobs ratio
    # token_level_logprobs= calculate_token_level_logprobs_ratio(teacher_log_probs_of_student, student_log_probs_of_student)
    # sentence_level_logprobs= calculate_sentence_level_logprobs_ratio(teacher_log_probs_of_student, student_log_probs_of_student)
    # all_outputs['teacher_student_token_log_prob_ratio']=token_level_logprobs  
    # all_outputs['teacher_student_sent_log_prob_ratio']=sentence_level_logprobs

    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if there is a directory part in the path
        os.makedirs(output_dir, exist_ok=True)  # Creates directory if it doesn't exist

    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=4)
    
    # Clear all remaining memory
    del data_student, data_tokenized, data_teacher, tokenizer, model
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
    parser.add_argument("--tokenized_prompt_path", type=str, default=None)
    parser.add_argument("--student_generation_path", type=str, default=None) 
    parser.add_argument("--teacher_generation_path", type=str, default=None)
    parser.add_argument("--student_logprobs_path", type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--torch_dtype', type=str, default='bfloat16')
    args = parser.parse_args() 

    get_logprobs(
        model_path=args.model_path,
        tokenized_prompt_path=args.tokenized_prompt_path,
        student_generation_path=args.student_generation_path,
        teacher_generation_path=args.teacher_generation_path,
        student_logprobs_path=args.student_logprobs_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        torch_dtype=args.torch_dtype
    )

if __name__=='__main__':
    main()