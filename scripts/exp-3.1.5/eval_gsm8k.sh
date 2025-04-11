#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.1.5/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/val/zero-shot/ \
    --exp_id exp-3.1.5 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.1.5/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/val/eight-shot/ \
    --exp_id exp-3.1.5 \
    --eval_id 2 \