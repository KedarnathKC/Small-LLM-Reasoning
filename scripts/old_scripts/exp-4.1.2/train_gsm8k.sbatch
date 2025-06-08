#!/bin/bash
#SBATCH --job-name=SFT_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-4.1.2/<jobname>_%A-%a.out 

bash scripts/exp-4.1.2/train_gsm8k.sh | tee ./outputs/exp-4.1.2/train.log
bash scripts/exp-4.1.2/eval_gsm8k.sh | tee ./outputs/exp-4.1.2/eval.log