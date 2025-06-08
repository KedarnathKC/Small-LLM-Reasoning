#!/bin/bash
#SBATCH --job-name=dpo_3B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-4:00:00
#SBATCH --output=outputs/exp-5/<jobname>_%A-%a.out 
#SBATCH --account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-5/dpo.sh | tee -a ./outputs/exp-5/dpo.log
bash scripts/exp-5/eval_gec.sh | tee -a ./outputs/exp-5/eval.log