#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft_vanilla.py \
    --model_name "meta-llama/Llama-3.2-3B" \
    --train_data_path "./datasets/gsm8k/train/" \
    --output_dir "./outputs/exp-3.2.2/checkpoints" \
    --add_special_tokens \
    --epochs 5 \
    --lr 1e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 500 

