#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.2.4/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/val/zero-shot/ \
    --output_path ./outputs/exp-3.2.4/val/0-shot \
    --exp_id exp-3.2.4 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.2.4/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/val/eight-shot/ \
    --output_path ./outputs/exp-3.2.4/val/8-shot \
    --exp_id exp-3.2.4 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.2.4/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/test/zero-shot/ \
    --output_path ./outputs/exp-3.2.4/test/0-shot \
    --exp_id exp-3.2.4 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-3.2.4/merged_checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/test/eight-shot/ \
    --output_path ./outputs/exp-3.2.4/test/8-shot \
    --exp_id exp-3.2.4 \
    --eval_id 1 \