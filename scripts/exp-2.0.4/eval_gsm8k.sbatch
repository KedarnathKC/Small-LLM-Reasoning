#!/bin/bash
#SBATCH --job-name=Eval_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-2.0.4/<jobname>_%A-%a.out 

bash scripts/exp-2.0.4/eval_gsm8k.sh | tee ./outputs/exp-2.0.4/eval.log

bash scripts/exp-2.0.5/eval_gsm8k.sh | tee ./outputs/exp-2.0.5/eval.log

bash scripts/exp-2.0.6/eval_gsm8k.sh | tee ./outputs/exp-2.0.6/eval.log

bash scripts/exp-2.0.7/eval_gsm8k.sh | tee ./outputs/exp-2.0.7/eval.log

bash scripts/exp-2.0.8/eval_gsm8k.sh | tee ./outputs/exp-2.0.8/eval.log

bash scripts/exp-2.0.9/eval_gsm8k.sh | tee ./outputs/exp-2.0.9/eval.log