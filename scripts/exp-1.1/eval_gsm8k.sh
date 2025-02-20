#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.1/checkpoints/checkpoint-1029 \
    --eval_data_path ./datasets/gsm8k/test \
    --exp_id exp-1.1 \
    --eval_id eval_2 \
    --prompting_strategy 8-shot
