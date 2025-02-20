#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/train_gsm8k.py \ 
    --model="meta-llama/Llama-3.2-1B-Instruct" \
    --data_path="./datasets/gsm8k/train/" \
    --exp_id="exp-1.1" \
    --batch_size=16