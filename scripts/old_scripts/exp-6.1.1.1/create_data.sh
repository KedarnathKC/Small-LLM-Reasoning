#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function preference_style_data_with_teacher_gen \
    --data_path ./datasets/neutralization/raw/feedback-100 \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-100/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.1.1/feedback-100/teacherlogprobs.json \
    --output_path  ./datasets/neutralization/generated/Llama-3.2-3B-Instruct/feedback-100/preference_style_data_with_teacher_gen \
    --formatting_func wnc \
    --input_col input 