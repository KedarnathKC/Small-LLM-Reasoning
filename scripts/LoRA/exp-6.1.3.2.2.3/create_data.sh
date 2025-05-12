#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function sft_data_with_teacher_gen \
    --data_path ./datasets/gec/feedback-1600 \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.2.2/feedback-1600/teacherlogprobs.json \
    --output_path  ./datasets/gec/sft/LLaMA8B/feedback-1600/sft_data_with_teacher_gen \
    --input_col input \
    --output_col output 