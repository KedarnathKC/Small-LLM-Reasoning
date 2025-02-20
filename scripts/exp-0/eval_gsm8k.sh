#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct  \
    --eval_data_path ./datasets/gsm8k/test \
    --exp_id exp-0 \
    --eval_id eval_3 \
    --prompting_strategy 8-shot