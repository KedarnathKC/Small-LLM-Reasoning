#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --gt_data_path ./datasets/neutralization/raw/feedback-400/ \
    --tokenized_prompt_path ./datasets/neutralization/tokenized/Llama-3.2-3B-Instruct/feedback-400/0-shot/ \
    --student_generation_path ./outputs/exp-1/feedback-400/0-shot/eval_1/generated_outputs.json  \
    --output_path ./outputs/exp-3/feedback-400/logprobs.json \
    --answer_col edits \
    --batch_size 16 