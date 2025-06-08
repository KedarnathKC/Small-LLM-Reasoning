#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function dpo_data_with_teacher_gen \
    --data_path ./datasets/gec/feedback-1600 \
    --teacher_data_path ./outputs/exp-5.1.2/feedback-1600/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.1.2/feedback-1600/teacherlogprobs.json \
    --output_path  ./datasets/gec/dpo/LLaMA3B/feedback-1600/dpo_data_with_teacher_gen \
    --formatting_func gec \
    --input_col input 