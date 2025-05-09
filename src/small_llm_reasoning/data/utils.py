import re
import random
import numpy as np
from datasets import load_from_disk, Dataset, load_dataset, concatenate_datasets

def formatting_prompt_func(questions):
    final_prompts=[]
    for question in questions:
        prompt = f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        final_prompts.append(prompt)
    return final_prompts

def get_prob(teacher_log_prob):
    teacher_logprob=[]
    for i in range(len(teacher_log_prob)):
        teacher_log_probs=np.array(teacher_log_prob[i])
        teacher_logprob.append(np.mean(teacher_log_probs))
    teacher_prob=np.exp(teacher_logprob)
    return teacher_prob

def get_log_prob_ratio(teacher_log_prob, student_log_prob):
    tr_stu_logprob=[]
    student_logprob=[]
    teacher_logprob=[]
    for i in range(len(teacher_log_prob)):
        student_log_probs=np.array(student_log_prob[i])
        teacher_log_probs=np.array(teacher_log_prob[i])
        student_logprob.append(np.mean(student_log_probs))
        teacher_logprob.append(np.mean(teacher_log_probs))
        tr_stu_logprob.append(
            np.subtract(
                np.mean(teacher_log_probs),
                np.mean(student_log_probs)
            )
        )
    return tr_stu_logprob

def create_sft_data_with_teacher_gen(data_path, teacher_data_path, student_data_path, output_path, input_col, output_col, remove_incorrects, incorrect_threshold=0):
    data = load_from_disk(data_path)
    teacher_data = load_dataset('json',data_files=teacher_data_path)['train']
    student_data = load_dataset('json',data_files=student_data_path)['train']

    tr_stu_logprob_ratio=get_log_prob_ratio(student_data['teacher_log_probs'],student_data['student_log_probs'])
    teacher_prob= get_prob(student_data['teacher_log_probs'])
   
    teacher_answers=[]
    teacher_rationale=[]
    teacher_scores=[]

    for i in range(teacher_data.num_rows):
        teacher_answers.append(teacher_data['model_answer'][i]) 
        teacher_rationale.append(teacher_data['model_rationale'][i])
        teacher_scores.append(teacher_data['score'][i])

    questions=data[input_col]
    new_data = {
        input_col: questions,
        output_col: teacher_answers,
        'rationale': teacher_rationale,
        'logprob_ratio':tr_stu_logprob_ratio,
        'tr_prob':teacher_prob,
        'tr_score': teacher_scores
    }

    data= Dataset.from_dict(new_data)

    if remove_incorrects:
        # Dont use >=, as in gsm8k where score is 0 or 1, and incorrect_threshold will be 0.
        data= data.filter(lambda x: x['tr_score']>incorrect_threshold)
    
    # Save the data
    data.save_to_disk(output_path)
    print(f'Created the dataset for SFT using teacher outputs as output for the given prompt, using all data with remove_incorrects={remove_incorrects}')
    print(f'Saved at: {output_path}')
    return 

def create_preference_data_with_teacher_gen(data_path, teacher_data_path, student_data_path, output_path, remove_incorrects=True):
    '''
        Function that converts the given dataset into preference dataset with additional columns for future filtering.
    '''
    data = load_from_disk(data_path)
    teacher_data = load_dataset('json',data_files=teacher_data_path)['train']
    student_data = load_dataset('json',data_files=student_data_path)['train']

    teacher_prob= get_prob(student_data['teacher_log_probs'])

    prompt= formatting_prompt_func(data['question'])
    chosen= [tr_output[0] for tr_output in teacher_data['output']] 
    rejected= student_data['student_reasoning']

    new_data = {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'tr_prob':teacher_prob,
        'tr_answer': teacher_data['model_answer'],
        'stu_answer': student_data['student_answer'],
        'tr_score': teacher_data['score']
        }

    preference_data= Dataset.from_dict(new_data)
    if remove_incorrects:
        preference_data= preference_data.filter(lambda x: x['tr_score']==1)
    
    preference_data = preference_data.map(lambda example: {"tr=stu": example["tr_answer"] == example["stu_answer"]})
    preference_data=preference_data.remove_columns(['tr_answer', 'stu_answer'])
    preference_data.save_to_disk(output_path)

    print(f'Created the prefernce dataset for DPO using teacher outputs as chosen, student outputs as rejected using all data with remove_incorrects={remove_incorrects}')
    print(f'Saved at: {output_path}')
    return

def create_preference_data_with_teacher_prob(data_path, teacher_data_path, student_data_path, output_path, threshold=0.6, remove_incorrects=True):
    data = load_from_disk(data_path)
    teacher_data = load_dataset('json',data_files=teacher_data_path)['train']
    student_data = load_dataset('json',data_files=student_data_path)['train']

    teacher_prob= get_prob(student_data['teacher_log_probs'])

    prompt= formatting_prompt_func(data['question'])
    chosen= [tr_output[0] for tr_output in teacher_data['output']] 
    rejected= student_data['student_reasoning']

    new_data = {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'tr_prob':teacher_prob,
        'tr_score': teacher_data['score']
        }
        
    preference_data= Dataset.from_dict(new_data)
    if remove_incorrects:
        preference_data= preference_data.filter(lambda x: x['tr_score']==1)
        output_path+='_remove_incorrects'
    preference_data= preference_data.filter(lambda x: x['tr_prob']<=threshold)
    preference_data=preference_data.remove_columns(['tr_score', 'tr_prob'])
    preference_data.save_to_disk(output_path)
    print('Created the prefernce dataset for DPO using teacher outputs as chosen, student outputs as rejected where teacher-probability on student generation is lesser than 0.6.')
    print(f'Saved at: {output_path}')
    return

def create_preference_data_with_tr_stu_correctness(data_path, teacher_data_path, student_data_path, output_path, remove_incorrects=True):
    data = load_from_disk(data_path)
    teacher_data = load_dataset('json',data_files=teacher_data_path)['train']
    student_data = load_dataset('json',data_files=student_data_path)['train']

    prompt= formatting_prompt_func(data['question'])
    chosen= [tr_output[0] for tr_output in teacher_data['output']] 
    rejected= student_data['student_reasoning']
    

    new_data = {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'tr_answer': teacher_data['model_answer'],
        'stu_answer': student_data['student_answer'],
        'tr_score': teacher_data['score'],
    }
    
    preference_data= Dataset.from_dict(new_data)
    if remove_incorrects:
        preference_data= preference_data.filter(lambda x: x['tr_score']==1)
        output_path+='_remove_incorrects'
    preference_data= preference_data.filter(lambda x: x['tr_answer']!=x['stu_answer'])
    preference_data=preference_data.remove_columns(['tr_score', 'tr_answer', 'stu_answer'])
    preference_data.save_to_disk(output_path)

    print('Created the prefernce dataset for DPO using teacher outputs as chosen, student outputs as rejected where teacher-answer != student-answer')
    print(f'Saved at: {output_path}')
    return

def create_preference_data_with_teacher_gen_by_sampling(data_path, output_path, threshold_col, sampling_ratio, threshold, remove_incorrects=True, seed=42):
    data = load_from_disk(data_path)

    rng = random.Random(seed)

    total_size = len(data)
    size_below = int(total_size * sampling_ratio)
    size_above = total_size - size_below

    below_thresh = data.filter(lambda example: example[threshold_col] < threshold)
    above_thresh = data.filter(lambda example: example[threshold_col] >= threshold)
    print(f'below threshold:{below_thresh.num_rows}')
    print(f'above threshold:{above_thresh.num_rows}')

    def upsample(ds, target_size):
        if len(ds) == 0:
            return ds  # Avoid divide-by-zero
        indices = [rng.randint(0, len(ds) - 1) for _ in range(target_size)]
        return ds.select(indices)

    sampled_below = upsample(below_thresh, size_below)
    sampled_above = upsample(above_thresh, size_above)

    data = concatenate_datasets([sampled_below, sampled_above])
    data =data.shuffle(seed=seed)

    data.save_to_disk(output_path)

    print(f'Created the prefernce dataset for DPO using teacher outputs as chosen, student outputs as rejected by thresholding on {threshold_col} < {threshold}.')
    print(f'Sampling was done with sampling_ratio={sampling_ratio}')
    print(f'Saved at: {output_path}')
    return

