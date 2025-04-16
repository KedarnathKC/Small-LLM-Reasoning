import argparse
import sys
import os


from small_llm_reasoning.data.utils import *

def dataset_exists(dataset_name):
    dir_path_os=f'./datasets/gsm8k/{dataset_name}'
    print(f'Checking directory: {dir_path_os}')
    if os.path.isdir(dir_path_os):
        print(f"Directory '{dir_path_os}' exists.")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Create datasets')
    parser.add_argument('--function', type=str, required=True,
                      choices=[
                        'dpo_data_with_teacher_gen', 
                        'dpo_data_tr_prob_threshold', 
                        'dpo_data_tr_stu_correctness',
                        'dpo_data_with_teacher_gen_by_sampling'
                      ],
                      help='Function to call: teacher_prob or tr_stu_correctness')
    parser.add_argument('--data_path', type=str, help='Path to the main data file')
    parser.add_argument('--teacher_data_path', type=str, help='Path to the teacher data JSON file')
    parser.add_argument('--student_data_path', type=str, help='Path to the student data JSON file')
    parser.add_argument('--output_path', type=str, help='Path where the output dataset will be saved')
    parser.add_argument('--threshold_col', type=str, help='Column to threshold on')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold to use one the smapling column')
    parser.add_argument('--sampling_ratio', type=float, default=0.9, help='ratio for sampling')
    parser.add_argument('--remove_incorrects', action='store_true', help='Remove incorrect answers from the dataset')
    
    args = parser.parse_args()

    if args.function == 'dpo_data_with_teacher_gen':
        # Check if the dataset exists by checking whether the directory exists
        if dataset_exists('dpo/'+args.function):
            print(f'Dataset {args.function} exists.')
            return
        create_preference_data_with_teacher_gen(
            data_path=args.data_path, 
            teacher_data_path=args.teacher_data_path, 
            student_data_path=args.student_data_path, 
            output_path=args.output_path, 
            remove_incorrects=args.remove_incorrects
        )
    elif args.function =='dpo_data_with_teacher_gen_by_sampling':
        if dataset_exists('dpo/'+args.function):
            print(f'Dataset {args.function} exists.')
            return
        create_preference_data_with_teacher_gen_by_sampling(
            data_path=args.data_path, 
            output_path=args.output_path, 
            threshold_col=args.threshold_col, 
            sampling_ratio=args.sampling_ratio, 
            threshold=args.threshold, 
            remove_incorrects=args.remove_incorrects, 
            seed=42
        )
    elif args.function == 'dpo_data_tr_prob_threshold':
        # Check if the dataset exists by checking whether the directory exists
        if dataset_exists('dpo/'+args.function):
            print(f'Dataset {args.function} exists.')
            return
        create_preference_data_with_teacher_prob(
            data_path=args.data_path,
            teacher_data_path=args.teacher_data_path,
            student_data_path=args.student_data_path,
            output_path=args.output_path,
            threshold=args.threshold,
            remove_incorrects=args.remove_incorrects
        )
    elif args.function == 'dpo_data_tr_stu_correctness':
        # Check if the dataset exists by checking whether the directory exists
        if dataset_exists('dpo/'+args.function):
            print(f'Dataset {args.function} exists already.')
            return
        create_preference_data_with_tr_stu_correctness(
            data_path=args.data_path,
            teacher_data_path=args.teacher_data_path,
            student_data_path=args.student_data_path,
            output_path=args.output_path,
            remove_incorrects=args.remove_incorrects
        )

if __name__ == '__main__':
    main()