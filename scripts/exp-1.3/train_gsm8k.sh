#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/train_gsm8k.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --train_data_path "./datasets/gsm8k/train/" \
    --output_dir "./outputs/exp-1.3/checkpoints" \
    --epochs 5 \
    --lr 3e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 500 