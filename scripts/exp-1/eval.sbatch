#!/bin/bash
#SBATCH --job-name=Eval_Neu_3B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-01:00:00
#SBATCH --output=outputs/exp-1/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-1/eval.sh | tee -a ./outputs/exp-1/eval.log
