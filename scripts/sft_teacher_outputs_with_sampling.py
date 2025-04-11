import os
import random
import argparse
import numpy as np
from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from sft import finetune

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

def create_data_from_teacher_gen_with_sampling(data, teacher_data, student_data, remove_incorrects, sampling_ratio, seed=42):
    tr_stu_logprob_ratio=get_log_prob_ratio(student_data['teacher_log_probs'],student_data['student_log_probs'])
    threshold=np.median(tr_stu_logprob_ratio)
    print(f'Median teacher-student-logprob-ratio: {threshold}')
    teacher_answers=[]
    teacher_scores=[]

    for i in range(teacher_data.num_rows):
        teacher_answers.append(teacher_data['output'][i][0])
        teacher_scores.append(teacher_data['score'][i])
    questions=data['question']
    new_data = {
        'question': questions,
        'answer': teacher_answers,
        'score': teacher_scores,
        'logprob_ratio':tr_stu_logprob_ratio
    }
    
    data= Dataset.from_dict(new_data)

    if remove_incorrects:
        data= data.filter(lambda x: x['score']==1)
    print(f'After removing incorrects from teacher:{data.num_rows}')
    
    rng = random.Random(seed)

    total_size = len(data)
    size_below = int(total_size * sampling_ratio)
    size_above = total_size - size_below

    below_thresh = data.filter(lambda example: example['logprob_ratio'] < threshold)
    above_thresh = data.filter(lambda example: example['logprob_ratio'] >= threshold)
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
    return data.shuffle(seed=seed)

def start_finetuning(model_name, train_data_path, teacher_data_path, student_data_path, sampling_ratio, remove_incorrect, response_template, output_dir, add_special_tokens, lora, epochs, lr, lr_scheduler_type, warmup, weight_decay, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    print('Loading data...')
    train_data=load_from_disk(train_data_path)
    teacher_data=load_dataset('json',data_files=teacher_data_path)['train']
    student_data=load_dataset('json',data_files=student_data_path)['train']
    print('Data loaded')

    train_data= create_data_from_teacher_gen_with_sampling(
        data=train_data, 
        teacher_data=teacher_data, 
        student_data=student_data, 
        remove_incorrects=remove_incorrect, 
        sampling_ratio=sampling_ratio
    )
    print('Teacher Generations replaced as answers')

    if remove_incorrect:
        print(f'Data size after removing teacher incorrets: {train_data.num_rows}')
    print(train_data['question'][0])
    print(train_data['answer'][0])

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
    parser.add_argument("--student_data_path", type=str, default=None)
    parser.add_argument('--sampling_ratio', type=float, default=0)
    parser.add_argument('--response_template', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--teacher_data_path', type=str, default=None)
    parser.add_argument("--remove_incorrect", action="store_true", help="Set this flag to true", default=False)
    parser.add_argument("--add_special_tokens", action='store_true', help='Set this flag to true', default=True)
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
        student_data_path=args.student_data_path,
        sampling_ratio=args.sampling_ratio,
        response_template=args.response_template,
        output_dir=args.output_dir, 
        teacher_data_path=args.teacher_data_path,
        remove_incorrect=args.remove_incorrect,
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