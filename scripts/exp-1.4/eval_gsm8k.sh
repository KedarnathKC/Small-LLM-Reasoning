#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path ./datasets/gsm8k/test \
    --exp_id exp-1.4 \
    --eval_id eval_2 \
    --lora \
    --lora_name Lora_Finetuned_3B \
    --lora_int_id 2\
    --lora_path ./outputs/exp-1.4/checkpoints/checkpoint-2055 \
    --prompting_strategy 8-shot
