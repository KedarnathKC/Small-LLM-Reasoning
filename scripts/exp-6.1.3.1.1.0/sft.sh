#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --train_data_path ./datasets/neutralization/sft/LLaMA8B/sft_data_with_teacher_gen \
    --output_dir ./outputs/exp-6.1.3.1.1.0/checkpoints/ \
    --formatting_func wnc \
    --add_special_tokens \
    --epochs 5 \
    --lr 5e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16\
    --max_seq_length 500