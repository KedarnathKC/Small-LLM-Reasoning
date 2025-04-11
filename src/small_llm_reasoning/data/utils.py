import re
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

def create_data_from_teacher_gen(data, teacher_data, remove_incorrects):
    teacher_answers=[]
    teacher_scores=[]

    for i in range(teacher_data.num_rows):
        teacher_answers.append(teacher_data['output'][i][0])
        teacher_scores.append(teacher_data['score'][i])


    questions=data['question']
    new_data = {
        'question': questions,
        'answer': teacher_answers,
        'score': teacher_scores
    }

    data= Dataset.from_dict(new_data)

    if remove_incorrects:
        data= data.filter(lambda x: x['score']==1)
    
    data=data.remove_columns('score')
    return data

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
    preference_data= preference_data.filter(lambda x: x['tr_answer']!=x['stu_answer'])
    preference_data=preference_data.remove_columns(['tr_score', 'tr_answer', 'stu_answer'])
    preference_data.save_to_disk(output_path)

    print('Created the prefernce dataset for DPO using teacher outputs as chosen, student outputs as rejected where teacher-answer != student-answer')
    print(f'Saved at: {output_path}')
    return

    