#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH --output=outputs/exp-3.1.1/<jobname>_%A-%a.out 

bash scripts/exp-3.1.1/train_gsm8k.sh | tee ./outputs/exp-3.1.1/train.log
bash scripts/exp-3.1.1/eval_gsm8k.sh | tee ./outputs/exp-3.1.1/eval.log

bash scripts/exp-3.1.2/train_gsm8k.sh | tee ./outputs/exp-3.1.2/train.log
bash scripts/exp-3.1.2/eval_gsm8k.sh | tee ./outputs/exp-3.1.2/eval.log

bash scripts/exp-3.1.3/train_gsm8k.sh | tee ./outputs/exp-3.1.3/train.log
bash scripts/exp-3.1.3/eval_gsm8k.sh | tee ./outputs/exp-3.1.3/eval.log