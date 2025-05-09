#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function sft_data_with_teacher_gen \
    --data_path ./datasets/neutralization/feedback-400 \
    --teacher_data_path ./outputs/exp-5.1.1/feedback-400/0-shot/eval_1/generated_outputs.json \
    --student_data_path ./outputs/exp-5.2.1.1/feedback-400/teacherlogprobs.json \
    --output_path  ./datasets/neutralization/sft/LLaMA3B/feedback-400/sft_data_with_teacher_gen \
    --input_col input \
    --output_col edits 