#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --gt_data_path ./datasets/gsm8k/feedback/ \
    --tokenized_data_path ./datasets/gsm8k/tokenized/LLaMA8B-Instruct/feedback/zero-shot/ \
    --student_data_path ./outputs/exp-2.0.2/eval_1/generated_outputs.json \
    --batch_size 4 \
    --exp_id exp-2.1.2 \
    --eval_id 1