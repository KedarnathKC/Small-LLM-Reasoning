#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_data_path ./datasets/gsm8k/tokenized/LLaMA8B-Instruct/feedback/zero-shot/ \
    --student_data_path ./outputs/exp-2.0.2/eval_1/generated_outputs.json \
    --teacher_data_path ./outputs/exp-2.0.3/eval_1/generated_outputs.json \
    --batch_size 1 \
    --exp_id exp-2.1.2 \
    --eval_id 1