#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft_teacher_outputs_with_sampling.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --train_data_path "./datasets/gsm8k/feedback/" \
    --teacher_data_path "./outputs/exp-2.0.3/eval_1/generated_outputs.json" \
    --student_data_path "./outputs/exp-2.1.1/eval_1/logprobs1.json" \
    --remove_incorrect \
    --sampling_ratio 0.9 \
    --output_dir "./outputs/exp-4.2.1/checkpoints" \
    --add_special_tokens \
    --epochs 5 \
    --lr 5e-6 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 500 