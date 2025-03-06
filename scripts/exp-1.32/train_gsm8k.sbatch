#!/bin/bash
#SBATCH --job-name=SFT_3B_Instruct
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-1.32/<jobname>_%A-%a.out 

# bash scripts/exp-1.27/train_gsm8k.sh | tee ./outputs/exp-1.27/train.log
# bash scripts/exp-1.27/eval_gsm8k.sh | tee ./outputs/exp-1.27/eval_train.log

# # bash scripts/exp-1.28/train_gsm8k.sh | tee ./outputs/exp-1.28/train.log
# bash scripts/exp-1.28/eval_gsm8k.sh | tee ./outputs/exp-1.28/eval_train.log

# # bash scripts/exp-1.29/train_gsm8k.sh | tee ./outputs/exp-1.29/train.log
# bash scripts/exp-1.29/eval_gsm8k.sh | tee ./outputs/exp-1.29/eval_train.log

# # bash scripts/exp-1.30/train_gsm8k.sh | tee ./outputs/exp-1.30/train.log
# bash scripts/exp-1.30/eval_gsm8k.sh | tee ./outputs/exp-1.30/eval_train.log

# # bash scripts/exp-1.31/train_gsm8k.sh | tee ./outputs/exp-1.31/train.log
# bash scripts/exp-1.31/eval_gsm8k.sh | tee ./outputs/exp-1.31/eval_train.log

# # bash scripts/exp-1.32/train_gsm8k.sh | tee ./outputs/exp-1.32/train.log
bash scripts/exp-1.32/eval_gsm8k.sh | tee ./outputs/exp-1.32/eval_train.log

