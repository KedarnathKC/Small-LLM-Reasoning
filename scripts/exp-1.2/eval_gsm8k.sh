#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --eval_data_path ./datasets/gsm8k/test \
    --exp_id exp-1.2 \
    --eval_id eval_1 \
    --lora \
    --lora_name Lora_Finetuned_1B \
    --lora_int_id 1\
    --lora_path ./outputs/exp-1.2/checkpoints/checkpoint-1029
