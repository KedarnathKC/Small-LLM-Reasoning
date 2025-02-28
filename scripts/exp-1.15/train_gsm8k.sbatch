#!/bin/bash
#SBATCH --job-name=SFT_3B_Instruct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-1.15/<jobname>_%A-%a.out 

bash scripts/exp-1.15/train_gsm8k.sh | tee ./outputs/exp-1.15/train.log
bash scripts/exp-1.15/eval_gsm8k.sh | tee ./outputs/exp-1.15/eval.log

bash scripts/exp-1.16/train_gsm8k.sh | tee ./outputs/exp-1.16/train.log
bash scripts/exp-1.16/eval_gsm8k.sh | tee ./outputs/exp-1.16/eval.log

bash scripts/exp-1.17/train_gsm8k.sh | tee ./outputs/exp-1.17/train.log
bash scripts/exp-1.17/eval_gsm8k.sh | tee ./outputs/exp-1.17/eval.log

bash scripts/exp-1.18/train_gsm8k.sh | tee ./outputs/exp-1.18/train.log
bash scripts/exp-1.18/eval_gsm8k.sh | tee ./outputs/exp-1.18/eval.log

bash scripts/exp-1.19/train_gsm8k.sh | tee ./outputs/exp-1.19/train.log
bash scripts/exp-1.19/eval_gsm8k.sh | tee ./outputs/exp-1.19/eval.log

bash scripts/exp-1.20/train_gsm8k.sh | tee ./outputs/exp-1.20/train.log
bash scripts/exp-1.20/eval_gsm8k.sh | tee ./outputs/exp-1.20/eval.log

