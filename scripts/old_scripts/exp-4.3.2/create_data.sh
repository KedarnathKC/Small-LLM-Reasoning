#!/bin/bash
export NO_PROGRESS_BAR=true
hostname

python scripts/create_data.py \
    --function dpo_data_with_teacher_gen_by_sampling \
    --data_path ./datasets/gsm8k/dpo/dpo_data_with_teacher_gen/ \
    --output_path ./datasets/gsm8k/dpo/dpo_data_with_teacher_gen_sampling_tr_gen_neq_stu_gen/ \
    --threshold_col 'tr=stu' \
    --threshold 1 \
    --sampling_ratio 0.9 
