#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-3B  \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/val/without-chat-template/zero-shot/ \
    --exp_id exp-3.0 \
    --eval_id 3

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-3B  \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/val/without-chat-template/eight-shot/ \
    --exp_id exp-3.0 \
    --eval_id 4