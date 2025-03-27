#!/bin/bash
#SBATCH --job-name=stu_logprobs
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-2:00:00
#SBATCH --output=outputs/exp-2.1.2/<jobname>_%A-%a.out 

bash scripts/exp-2.1.2/studentlogprobs.sh 