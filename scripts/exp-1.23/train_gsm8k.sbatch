#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-06:00:00
#SBATCH --output=outputs/exp-1.23/<jobname>_%A-%a.out 

# bash scripts/exp-1.21/train_gsm8k.sh | tee ./outputs/exp-1.21/train.log
# bash scripts/exp-1.21/eval_gsm8k.sh | tee ./outputs/exp-1.21/eval_train.log

# # bash scripts/exp-1.22/train_gsm8k.sh | tee ./outputs/exp-1.22/train.log
# bash scripts/exp-1.22/eval_gsm8k.sh | tee ./outputs/exp-1.22/eval_train.log

# # bash scripts/exp-1.23/train_gsm8k.sh | tee ./outputs/exp-1.23/train.log
bash scripts/exp-1.23/eval_gsm8k.sh | tee ./outputs/exp-1.23/eval_train.log

# # bash scripts/exp-1.24/train_gsm8k.sh | tee ./outputs/exp-1.24/train.log
# bash scripts/exp-1.24/eval_gsm8k.sh | tee ./outputs/exp-1.24/eval_train.log

# # bash scripts/exp-1.25/train_gsm8k.sh | tee ./outputs/exp-1.25/train.log
# bash scripts/exp-1.25/eval_gsm8k.sh | tee ./outputs/exp-1.25/eval_train.log

# # bash scripts/exp-1.26/train_gsm8k.sh | tee ./outputs/exp-1.26/train.log
# bash scripts/exp-1.26/eval_gsm8k.sh | tee ./outputs/exp-1.26/eval_train.log