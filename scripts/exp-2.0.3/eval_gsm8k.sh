#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/gsm8k/tokenized/LLaMA70B/feedback/zero-shot/ \
    --log_probs 1 \
    --n_gpus 4 \
    --exp_id exp-2.0.3 \
    --eval_id 1 