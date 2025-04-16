#!/bin/bash
#SBATCH --job-name=eval_SFT
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-3.2.6/<job-name>_%A-%a.out 

bash scripts/exp-3.2.6/eval_gsm8k.sh | tee ./outputs/exp-3.2.6/eval.log