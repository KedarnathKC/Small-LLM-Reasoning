#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/train_gsm8k.py \ 
    --model="meta-llama/Llama-3.2-3B-Instruct" \
    --data_path="./datasets/gsm8k/train/" \
    --exp_id="exp-1.4" \
    --lora \
    --batch_size=8