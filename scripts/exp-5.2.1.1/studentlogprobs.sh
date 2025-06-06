#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

# python scripts/hf_logprobs_student.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --gt_data_path ./datasets/neutralization/feedback/ \
#     --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback/0-shot/ \
#     --student_data_path ./outputs/exp-5.0.1.1/feedback/0-shot/eval_1/generated_outputs.json \
#     --output_path ./outputs/exp-5.2.1.1/feedback/studentlogprobs.json \
#     --answer_col edits \
#     --batch_size 16 \

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --gt_data_path ./datasets/neutralization/feedback-100/ \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-100/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.1/feedback-100/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.1.1/feedback-100/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --gt_data_path ./datasets/neutralization/feedback-400/ \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-400/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.1/feedback-400/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.1.1/feedback-400/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --gt_data_path ./datasets/neutralization/feedback-1600/ \
    --tokenized_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-1600/0-shot/ \
    --student_data_path ./outputs/exp-5.0.1.1/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --output_path ./outputs/exp-5.2.1.1/feedback-1600/studentlogprobs.json \
    --answer_col edits \
    --batch_size 16 \