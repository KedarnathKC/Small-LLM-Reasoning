#!/bin/bash
#SBATCH --job-name=M1_3B_NEU_F1600
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-12:00:00
#SBATCH --output=outputs/exp-6.1.1.3/<jobname>_%A-%a.out 
#SBATCH --account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-6.1.1.3/create_data.sh | tee -a ./outputs/exp-6.1.1.3/data.log
bash scripts/exp-6.1.1.3/dpo.sh | tee -a ./outputs/exp-6.1.1.3/dpo.log
bash scripts/exp-6.1.1.3/eval_neutralization.sh | tee -a ./outputs/exp-6.1.1.3/eval.log