#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_gec.py \
    --model_path ./outputs/exp-6.1.2.1.2.3/checkpoints/ \
    --eval_data_path ./datasets/gec/tokenized/LLaMA3B-Instruct/val/0-shot/ \
    --output_path ./outputs/exp-6.1.2.1.2.3/val/0-shot \
    --m2_file_path ./datasets/gec/val/val.m2 \
    --reference_col edits \
    --n_gpus 1 \
    --top_p 1 \
    --top_k 1 \
    --eval_id 1 

python scripts/evaluation_gec.py \
    --model_path ./outputs/exp-6.1.2.1.2.3/checkpoints/ \
    --eval_data_path ./datasets/gec/tokenized/LLaMA3B-Instruct/test/0-shot/ \
    --output_path ./outputs/exp-6.1.2.1.2.3/test/0-shot \
    --m2_file_path ./datasets/gec/test/test.m2 \
    --reference_col edits \
    --n_gpus 1 \
    --top_p 1 \
    --top_k 1 \
    --eval_id 1 