#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.1/checkpoints/checkpoint-1715 \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA1B/test/eight-shot/ \
    --exp_id exp-1.1 \
    --eval_id eval_10 \