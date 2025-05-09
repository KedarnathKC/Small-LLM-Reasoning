#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/sft.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --train_data_path ./datasets/neutralization/sft/LLaMA3B/feedback-100/sft_data_with_teacher_gen \
    --output_dir ./outputs/exp-6.2.2.2.1.1/checkpoints/ \
    --formatting_func wnc \
    --add_special_tokens \
    --sample \
    --sampling_ratio 0.9 \
    --threshold_col logprob_ratio \
    --epochs 5 \
    --lora \
    --lr 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4\
    --max_seq_length 500

python src/small_llm_reasoning/utils/merge_lora_weights.py \
    --base_model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --checkpoints_path "./outputs/exp-6.2.2.2.1.1/checkpoints" \
    --save_path ./outputs/exp-6.2.2.2.1.1/merged_checkpoints \
    --torch_dtype bfloat16