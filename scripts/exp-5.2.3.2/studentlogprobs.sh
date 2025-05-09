#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --gt_data_path ./datasets/gec/feedback-100/ \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA1B-Instruct/feedback-100/0-shot/ \
    --student_data_path ./outputs/exp-5.0.3.2/feedback-100/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.3.2/feedback-100/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --gt_data_path ./datasets/gec/feedback-400/ \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA1B-Instruct/feedback-400/0-shot/ \
    --student_data_path ./outputs/exp-5.0.3.2/feedback-400/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.3.2/feedback-400/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --gt_data_path ./datasets/gec/feedback-1600/ \
    --tokenized_data_path ./datasets/gec/tokenized/LLaMA1B-Instruct/feedback-1600/0-shot/ \
    --student_data_path ./outputs/exp-5.0.3.2/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.3.2/feedback-1600/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \