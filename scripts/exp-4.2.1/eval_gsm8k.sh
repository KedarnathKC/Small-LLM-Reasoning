#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-4.2.1/checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/val/zero-shot/ \
    --output_path ./outputs/exp-4.2.1/val/0-shot \
    --exp_id exp-4.2.1 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-4.2.1/checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/val/eight-shot/ \
    --output_path ./outputs/exp-4.2.1/val/8-shot \
    --exp_id exp-4.2.1 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-4.2.1/checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/test/zero-shot/ \
    --output_path ./outputs/exp-4.2.1/test/0-shot \
    --exp_id exp-4.2.1 \
    --eval_id 1 \

python scripts/evaluation_gsm8k.py \
    --model_path "./outputs/exp-4.2.1/checkpoints" \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/test/eight-shot/ \
    --output_path ./outputs/exp-4.2.1/test/8-shot \
    --exp_id exp-4.2.1 \
    --eval_id 1 \