#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/Llama-3.3-70B-Instruct/feedback-400/0-shot/ \
    --output_path ./outputs/exp-2/feedback-400/0-shot \
    --evaluation_func neutralization \
    --input_col input \
    --reference_col edits \
    --eval_id 1 \
    --n_gpus 4