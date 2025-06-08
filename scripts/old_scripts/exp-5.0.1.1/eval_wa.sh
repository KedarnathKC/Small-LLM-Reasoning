#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

# python scripts/evaluation_wa.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/val/0-shot/ \
#     --output_path ./outputs/exp-5.0.1.1/val/0-shot \
#     --input_col input \
#     --reference_col edits \
#     --log_probs 1 \
#     --eval_id 1 \

# python scripts/evaluation_wa.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback/0-shot/ \
#     --output_path ./outputs/exp-5.0.1.1/feedback/0-shot \
#     --input_col input \
#     --reference_col edits \
#     --log_probs 1 \
#     --eval_id 1 \

# python scripts/evaluation_wa.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/test/0-shot/ \
#     --output_path ./outputs/exp-5.0.1.1/test/0-shot \
#     --input_col input \
#     --reference_col edits \
#     --log_probs 1 \
#     --eval_id 1 \

# python scripts/evaluation_wa.py \
#     --model_path meta-llama/Llama-3.2-3B-Instruct \
#     --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-100/0-shot/ \
#     --output_path ./outputs/exp-5.0.1.1/feedback-100/0-shot \
#     --input_col input \
#     --reference_col edits \
#     --log_probs 1 \
#     --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-400/0-shot/ \
    --output_path ./outputs/exp-5.0.1.1/feedback-400/0-shot \
    --input_col input \
    --reference_col edits \
    --log_probs 1 \
    --eval_id 1 \

python scripts/evaluation_wa.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path ./datasets/neutralization/tokenized/LLaMA3B-Instruct/feedback-1600/0-shot/ \
    --output_path ./outputs/exp-5.0.1.1/feedback-1600/0-shot \
    --input_col input \
    --reference_col edits \
    --log_probs 1 \
    --eval_id 1 \