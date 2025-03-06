#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-1.9/<jobname>_%A-%a.out 

bash scripts/exp-1.9/train_gsm8k.sh | tee ./outputs/exp-1.9/train.log
bash scripts/exp-1.9/eval_gsm8k.sh | tee ./outputs/exp-1.9/eval.log

bash scripts/exp-1.10/train_gsm8k.sh | tee ./outputs/exp-1.10/train.log
bash scripts/exp-1.10/eval_gsm8k.sh | tee ./outputs/exp-1.10/eval.log

bash scripts/exp-1.11/train_gsm8k.sh | tee ./outputs/exp-1.11/train.log
bash scripts/exp-1.11/eval_gsm8k.sh | tee ./outputs/exp-1.11/eval.log

bash scripts/exp-1.12/train_gsm8k.sh | tee ./outputs/exp-1.12/train.log
bash scripts/exp-1.12/eval_gsm8k.sh | tee ./outputs/exp-1.12/eval.log

bash scripts/exp-1.13/train_gsm8k.sh | tee ./outputs/exp-1.13/train.log
bash scripts/exp-1.13/eval_gsm8k.sh | tee ./outputs/exp-1.13/eval.log

bash scripts/exp-1.14/train_gsm8k.sh | tee ./outputs/exp-1.14/train.log
bash scripts/exp-1.14/eval_gsm8k.sh | tee ./outputs/exp-1.14/eval.log