#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function sft_data_with_teacher_gen \
    --data_path ./datasets/gec/feedback-100 \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-100/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.1.2/feedback-100/teacherlogprobs.json \
    --output_path  ./datasets/gec/sft/LLaMA3B/feedback-100/sft_data_with_teacher_gen \
    --input_col input \
    --output_col output 