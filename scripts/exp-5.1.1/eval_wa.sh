#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA70B-Instruct/val/0-shot/ \
    --output_path ./outputs/exp-5.1.1/val/0-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 4  \
    --log_probs 1 \
    --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA70B-Instruct/val/3-shot/ \
    --output_path ./outputs/exp-5.1.1/val/3-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 4  \
    --log_probs 1 \
    --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA70B-Instruct/feedback/0-shot/ \
    --output_path ./outputs/exp-5.1.1/feedback/0-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 4  \
    --log_probs 1 \
    --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA70B-Instruct/test/0-shot/ \
    --output_path ./outputs/exp-5.1.1/test/0-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 4  \
    --log_probs 1 \
    --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA70B-Instruct/test/3-shot/ \
    --output_path ./outputs/exp-5.1.1/test/3-shot \
    --input_col input \
    --reference_col edits \
    --n_gpus 4  \
    --log_probs 1 \
    --eval_id 1 \