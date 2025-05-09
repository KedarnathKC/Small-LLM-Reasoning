#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA3B-Instruct/feedback-100/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.2/feedback-100/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-100/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.1.2/feedback-100/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.1.2/feedback-100/teacherlogprobs.json \
    --batch_size 1 \

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA3B-Instruct/feedback-400/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.2/feedback-400/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-400/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.1.2/feedback-400/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.1.2/feedback-400/teacherlogprobs.json \
    --batch_size 1 \

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA3B-Instruct/feedback-1600/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.2/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-5.2.1.2/feedback-1600/studentlogprobs.json \
    --output_path ./outputs/exp-5.2.1.2/feedback-1600/teacherlogprobs.json \
    --batch_size 1 \