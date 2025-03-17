#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/get_logprobs.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --data_path ./outputs/exp-2.0.2/eval_1/generated_outputs.json \
    --batch_size 8 \
    --exp_id exp-2.1.2 \
    --eval_id 1
