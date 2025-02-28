#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.7/checkpoints/checkpoint-2740 \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B/test/eight-shot/ \
    --exp_id exp-1.7 \
    --eval_id eval_8 \