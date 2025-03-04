#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct  \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B/test/eight-shot/ \
    --exp_id exp-0 \
    --eval_id eval_4