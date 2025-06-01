#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gsm8k.py \
    --model_path /datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/  \
    --eval_data_path ./datasets/gsm8k/tokenized/Llama-3.3-70B-Instruct/test/8-shot/ \
    --output_path ./outputs/exp-0/test/8-shot/ \
    --n_gpus 4 \
    --eval_id 1