#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA8B-Instruct/feedback/zero-shot/ \
    --log_probs 1 \
    --n_gpus 1 \
    --exp_id exp-2.0.2 \
    --eval_id 1 