#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-3.1.4/<jobname>_%A-%a.out 

bash scripts/exp-3.1.4/train_gsm8k.sh | tee ./outputs/exp-3.1.4/train.log
bash scripts/exp-3.1.4/eval_gsm8k.sh | tee ./outputs/exp-3.1.4/eval.log

bash scripts/exp-3.1.5/train_gsm8k.sh | tee ./outputs/exp-3.1.5/train.log
bash scripts/exp-3.1.5/eval_gsm8k.sh | tee ./outputs/exp-3.1.5/eval.log

bash scripts/exp-3.1.6/train_gsm8k.sh | tee ./outputs/exp-3.1.6/train.log
bash scripts/exp-3.1.6/eval_gsm8k.sh | tee ./outputs/exp-3.1.6/eval.log