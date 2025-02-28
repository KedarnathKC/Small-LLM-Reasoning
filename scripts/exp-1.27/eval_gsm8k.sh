#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.27/checkpoints/\
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B/test/zero-shot/ \
    --exp_id exp-1.27 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.27/checkpoints/\
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B/test/eight-shot/ \
    --exp_id exp-1.27 \
    --eval_id 2 \