#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function dpo_data_tr_prob_threshold \
    --data_path ./datasets/gsm8k/feedback/ \
    --teacher_data_path ./outputs/exp-2.0.3/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-2.1.1/eval_1/logprobs1.json \
    --output_path ./datasets/gsm8k/dpo/dpo_data_tr_prob_threshold/ \
    --remove_incorrects \
    --threshold 0.6
