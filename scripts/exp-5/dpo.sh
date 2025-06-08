#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/dpo.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --train_data_path ./outputs/exp-3/feedback-400/logprobs.json \
    --output_dir ./outputs/exp-5/checkpoints/ \
    --max_steps 1000 \
    --lr 1e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 500