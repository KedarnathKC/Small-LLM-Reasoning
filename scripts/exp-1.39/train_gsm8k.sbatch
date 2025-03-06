#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-1.39/<jobname>_%A-%a.out 

bash scripts/exp-1.39/train_gsm8k.sh | tee ./outputs/exp-1.39/train.log
bash scripts/exp-1.39/eval_gsm8k.sh | tee ./outputs/exp-1.39/eval.log

bash scripts/exp-1.40/train_gsm8k.sh | tee ./outputs/exp-1.40/train.log
bash scripts/exp-1.40/eval_gsm8k.sh | tee ./outputs/exp-1.40/eval.log

bash scripts/exp-1.41/train_gsm8k.sh | tee ./outputs/exp-1.41/train.log
bash scripts/exp-1.41/eval_gsm8k.sh | tee ./outputs/exp-1.41/eval.log

bash scripts/exp-1.42/train_gsm8k.sh | tee ./outputs/exp-1.42/train.log
bash scripts/exp-1.42/eval_gsm8k.sh | tee ./outputs/exp-1.42/eval.log

bash scripts/exp-1.43/train_gsm8k.sh | tee ./outputs/exp-1.43/train.log
bash scripts/exp-1.43/eval_gsm8k.sh | tee ./outputs/exp-1.43/eval.log

bash scripts/exp-1.44/train_gsm8k.sh | tee ./outputs/exp-1.44/train.log
bash scripts/exp-1.44/eval_gsm8k.sh | tee ./outputs/exp-1.44/eval.log

