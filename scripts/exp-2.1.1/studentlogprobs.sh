#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_student.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --gt_data_path ./datasets/gsm8k/feedback/ \
    --tokenized_data_path ./datasets/gsm8k/tokenized/LLaMA3B-Instruct/feedback/zero-shot/ \
    --student_data_path ./outputs/exp-2.0.1/eval_1/generated_outputs.json \
    --batch_size 8 \
    --exp_id exp-2.1.1 \
    --eval_id 1