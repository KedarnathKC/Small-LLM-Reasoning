#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft_teacher_outputs.py \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --train_data_path "./datasets/gsm8k/feedback/" \
    --teacher_data_path "./outputs/exp-2.0.3/eval_1/generated_outputs.json" \
    --remove_incorrect \
    --output_dir "./outputs/exp-4.1.2/checkpoints" \
    --add_special_tokens \
    --lora \
    --epochs 5 \
    --lr 1e-5 \
    --lr_scheduler_type 'cosine' \
    --warmup 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 500 

python src/small_llm_reasoning/utils/merge_lora_weights.py \
    --base_model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --checkpoints_path "./outputs/exp-4.1.2/checkpoints" \
    --save_path "./outputs/exp-4.1.2/merged_checkpoints" \
    --torch_dtype bfloat16