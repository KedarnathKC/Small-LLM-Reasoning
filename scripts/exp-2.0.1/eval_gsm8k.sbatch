#!/bin/bash
#SBATCH --job-name=logprobs_3B
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-2.0.1/<jobname>_%A-%a.out 

bash scripts/exp-2.0.1/eval_gsm8k.sh