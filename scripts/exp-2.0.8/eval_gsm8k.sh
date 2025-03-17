#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.2-3B \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Pretrained/feedback/zero-shot/ \
    --n_gpus 1 \
    --exp_id exp-2.0.8 \
    --eval_id 1