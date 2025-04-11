import argparse
import sys
import os


from small_llm_reasoning.data.utils import (
    create_preference_data_with_teacher_prob,
    create_preference_data_with_tr_stu_correctness
)

def main():
    parser = argparse.ArgumentParser(description='Create datasets')
    parser.add_argument('--function', type=str, required=True,
                      choices=['dpo_data_tr_prob_threshold', 'dpo_data_tr_stu_correctness'],
                      help='Function to call: teacher_prob or tr_stu_correctness')
    parser.add_argument('--data_path', type=str, help='Path to the main data file')
    parser.add_argument('--teacher_data_path', type=str, help='Path to the teacher data JSON file')
    parser.add_argument('--student_data_path', type=str, help='Path to the student data JSON file')
    parser.add_argument('--output_path', type=str, help='Path where the output dataset will be saved')
    parser.add_argument('--remove_incorrects', action='store_true', help='Remove incorrect answers from the dataset')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for teacher probability (only for teacher_prob function)')
    
    args = parser.parse_args()

    if args.function == 'dpo_data_tr_prob_threshold':
        create_preference_data_with_teacher_prob(
            data_path=args.data_path,
            teacher_data_path=args.teacher_data_path,
            student_data_path=args.student_data_path,
            output_path=args.output_path,
            threshold=args.threshold,
            remove_incorrects=args.remove_incorrects
        )
    elif args.function == 'dpo_data_tr_stu_correctness':
        create_preference_data_with_tr_stu_correctness(
            data_path=args.data_path,
            teacher_data_path=args.teacher_data_path,
            student_data_path=args.student_data_path,
            output_path=args.output_path,
            remove_incorrects=args.remove_incorrects
        )

if __name__ == '__main__':
    main()