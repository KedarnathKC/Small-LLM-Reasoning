#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --train_data_path ./datasets/neutralization/feedback \
    --output_dir ./outputs/exp-6.1.3.2/checkpoints/ \
    --formatting_func wnc \
    --add_special_tokens \
    --epochs 5 \
    --lora \
    --lr 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 500

python src/small_llm_reasoning/utils/merge_lora_weights.py \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --checkpoints_path "./outputs/exp-6.1.3.2/checkpoints" \
    --save_path "./outputs/exp-6.1.3.2/merged_checkpoints" \
    --torch_dtype bfloat16