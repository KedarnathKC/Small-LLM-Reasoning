#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_wa.py \
    --model_path ./outputs/exp-6.1.2.2.1.0/merged_checkpoints/ \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/val/0-shot/ \
    --output_path ./outputs/exp-6.1.2.2.1.0/val/0-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 1 \
    --eval_id 1 

python scripts/evaluation_wa.py \
    --model_path ./outputs/exp-6.1.2.2.1.0/merged_checkpoints/ \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/test/0-shot/ \
    --output_path ./outputs/exp-6.1.2.2.1.0/test/0-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 1 \
    --eval_id 1 