#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --train_data_path ./datasets/neutralization/sft/LLaMA3B/feedback-1600/sft_data_with_teacher_gen \
    --output_dir ./outputs/exp-6.2.1.3/checkpoints/ \
    --formatting_func wnc \
    --add_special_tokens \
    --sample \
    --sampling_ratio 0.9 \
    --threshold_col logprob_ratio \
    --max_steps 1000 \
    --lr 5e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4\
    --max_seq_length 500
