#!/bin/bash
#SBATCH --job-name=Eval_1B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-3:00:00
#SBATCH --output=outputs/exp-0/<jobname>_%A-%a.out 

bash scripts/exp-0/eval_gsm8k.sh