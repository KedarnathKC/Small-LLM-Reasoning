#!/bin/bash
#SBATCH --job-name=logprobs
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH -t 0-12:00:00
#SBATCH --output=outputs/exp-2.1.1/<jobname>_%A-%a.out 

bash scripts/exp-2.1.1/logprobs.sh 
bash scripts/exp-2.1.2/logprobs.sh 