#!/bin/bash
#SBATCH --job-name=eval_SFT
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-4.2.1/<job-name>_%A-%a.out 

bash scripts/exp-4.2.1/eval_gsm8k.sh | tee ./outputs/exp-4.2.1/eval.log

bash scripts/exp-4.2.2/eval_gsm8k.sh | tee ./outputs/exp-4.2.2/eval.log