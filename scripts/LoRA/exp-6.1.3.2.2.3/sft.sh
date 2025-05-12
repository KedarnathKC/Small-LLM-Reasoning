#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --train_data_path ./datasets/gec/sft/LLaMA8B/feedback-1600/sft_data_with_teacher_gen \
    --output_dir ./outputs/exp-6.1.3.2.2.3/checkpoints/ \
    --formatting_func gec \
    --add_special_tokens \
    --max_steps 1000 \
    --lora \
    --lr 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4\
    --max_seq_length 500

python src/small_llm_reasoning/utils/merge_lora_weights.py \
    --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --checkpoints_path "./outputs/exp-6.1.3.2.2.3/checkpoints" \
    --save_path "./outputs/exp-6.1.3.2.2.3/merged_checkpoints" \
    --torch_dtype bfloat16