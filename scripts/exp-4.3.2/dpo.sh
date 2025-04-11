#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/dpo.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --train_data_path ./datasets/gsm8k/dpo/dpo_data_tr_stu_correctness/ \
    --output_dir ./outputs/trial/checkpoints/ \
    --add_special_tokens \
    --epochs 5 \
    --lr 1e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 500 