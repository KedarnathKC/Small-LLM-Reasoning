#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path ./outputs/exp-1.6/checkpoints/merged_model-3425 \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B/test/eight-shot/ \
    --exp_id exp-1.6 \
    --eval_id eval_10 \