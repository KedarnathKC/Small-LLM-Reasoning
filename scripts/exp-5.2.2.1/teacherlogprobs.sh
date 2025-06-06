#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

# python scripts/hf_logprobs_teacher.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA8B-Instruct/feedback/0-shot/ \
#     --student_data_path ./outputs/exp-5.0.2.1/feedback/0-shot/eval_1/generated_outputs.json \
#     --teacher_data_path ./outputs/exp-5.1.1/feedback/0-shot/eval_1/generated_outputs.json \
#     --student_logprobs_path ./outputs/exp-5.2.2.1/feedback/studentlogprobs.json \
#     --output_path ./outputs/exp-5.2.2.1/feedback/teacherlogprobs.json \
#     --batch_size 1 \

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA8B-Instruct/feedback-100/0-shot/ \
    --student_data_path ./outputs/exp-5.0.2.1/feedback-100/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-100/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.2.1/feedback-100/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.2.1/feedback-100/teacherlogprobs.json \
    --batch_size 1 \
    
python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA8B-Instruct/feedback-400/0-shot/ \
    --student_data_path ./outputs/exp-5.0.2.1/feedback-400/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-400/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.2.1/feedback-400/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.2.1/feedback-400/teacherlogprobs.json \
    --batch_size 1 \


python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA8B-Instruct/feedback-1600/0-shot/ \
    --student_data_path ./outputs/exp-5.0.2.1/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.2.1/feedback-1600/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.2.1/feedback-1600/teacherlogprobs.json \
    --batch_size 1 \