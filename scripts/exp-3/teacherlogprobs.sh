#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/hf_logprobs_teacher.py \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --tokenized_prompt_path ./datasets/neutralization/tokenized/Llama-3.2-3B-Instruct/feedback-400/0-shot/ \
    --student_generation_path ./outputs/exp-1/feedback-400/0-shot/eval_1/generated_outputs.json \
    --teacher_generation_path ./outputs/exp-2/feedback-400/0-shot/eval_1/generated_outputs.json \
    --student_logprobs_path ./outputs/exp-3/feedback-400/logprobs.json \
    --output_path ./outputs/exp-3/feedback-400/logprobs.json \
    --batch_size 1 