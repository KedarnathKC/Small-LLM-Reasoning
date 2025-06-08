#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation.py \
    --model_path ./outputs/exp-5/checkpoints/ \
    --eval_data_path ./datasets/neutralization/tokenized/Llama-3.2-3B-Instruct/feedback-400/0-shot/ \
    --output_path ./outputs/exp-5/feedback-400/0-shot \
    --evaluation_func neutralization \
    --input_col input \
    --reference_col edits \
    --eval_id 1 