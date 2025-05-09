#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/val/0-shot/ \
#     --output_path ./outputs/exp-5.1.2/val/0-shot \
#     --m2_file_path ./datasets/gec/val/val.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

python scripts/evaluation_gec.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/val/3-shot/ \
    --output_path ./outputs/exp-5.1.2/val/3-shot \
    --m2_file_path ./datasets/gec/val/val.m2 \
    --reference_col output \
    --n_gpus 4  \
    --top_p 1 \
    --top_k 1 \
    --eval_id 1 \

python scripts/evaluation_gec.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/test/0-shot/ \
    --output_path ./outputs/exp-5.1.2/test/0-shot \
    --m2_file_path ./datasets/gec/test/test.m2 \
    --reference_col output \
    --n_gpus 4  \
    --top_p 1 \
    --top_k 1 \
    --eval_id 1 \

python scripts/evaluation_gec.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/test/3-shot/ \
    --output_path ./outputs/exp-5.1.2/test/3-shot \
    --m2_file_path ./datasets/gec/test/test.m2 \
    --reference_col output \
    --n_gpus 4  \
    --top_p 1 \
    --top_k 1 \
    --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-100/0-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-100/0-shot \
#     --m2_file_path ./datasets/gec/feedback-100/feedback-100.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-100/3-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-100/3-shot \
#     --m2_file_path ./datasets/gec/feedback-100/feedback-100.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-400/0-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-400/0-shot \
#     --m2_file_path ./datasets/gec/feedback-400/feedback-400.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-400/3-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-400/3-shot \
#     --reference_col output \
#     --m2_file_path ./datasets/gec/feedback-400/feedback-400.m2 \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-1600/0-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-1600/0-shot \
#     --m2_file_path ./datasets/gec/feedback-1600/feedback-1600.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \

# python scripts/evaluation_gec.py \
#     --model_path meta-llama/Llama-3.3-70B-Instruct \
#     --eval_data_path ./datasets/gec/tokenized/LLaMA70B-Instruct/feedback-1600/3-shot/ \
#     --output_path ./outputs/exp-5.1.2/feedback-1600/3-shot \
#     --m2_file_path ./datasets/gec/feedback-1600/feedback-1600.m2 \
#     --reference_col output \
#     --n_gpus 4  \
#     --top_p 1 \
#     --top_k 1 \
#     --eval_id 1 \