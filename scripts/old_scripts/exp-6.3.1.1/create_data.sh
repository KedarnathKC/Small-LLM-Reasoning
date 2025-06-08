#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function dpo_data_with_teacher_gen \
    --data_path ./datasets/neutralization/feedback-100 \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-100/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.1.1/feedback-100/teacherlogprobs.json \
    --output_path  ./datasets/neutralization/dpo/LLaMA3B/feedback-100/dpo_data_with_teacher_gen \
    --formatting_func wnc \
    --input_col input 