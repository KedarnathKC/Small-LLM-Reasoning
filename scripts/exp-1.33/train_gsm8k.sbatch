#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-1.33/<jobname>_%A-%a.out 

bash scripts/exp-1.33/train_gsm8k.sh | tee ./outputs/exp-1.33/train.log
bash scripts/exp-1.33/eval_gsm8k.sh | tee ./outputs/exp-1.33/eval.log

bash scripts/exp-1.34/train_gsm8k.sh | tee ./outputs/exp-1.34/train.log
bash scripts/exp-1.34/eval_gsm8k.sh | tee ./outputs/exp-1.34/eval.log

bash scripts/exp-1.35/train_gsm8k.sh | tee ./outputs/exp-1.35/train.log
bash scripts/exp-1.35/eval_gsm8k.sh | tee ./outputs/exp-1.35/eval.log

bash scripts/exp-1.36/train_gsm8k.sh | tee ./outputs/exp-1.36/train.log
bash scripts/exp-1.36/eval_gsm8k.sh | tee ./outputs/exp-1.36/eval.log

bash scripts/exp-1.37/train_gsm8k.sh | tee ./outputs/exp-1.37/train.log
bash scripts/exp-1.37/eval_gsm8k.sh | tee ./outputs/exp-1.37/eval.log

bash scripts/exp-1.38/train_gsm8k.sh | tee ./outputs/exp-1.38/train.log
bash scripts/exp-1.38/eval_gsm8k.sh | tee ./outputs/exp-1.38/eval.log