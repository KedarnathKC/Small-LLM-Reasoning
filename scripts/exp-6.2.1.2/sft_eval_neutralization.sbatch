#!/bin/bash
#SBATCH --job-name=M2_3B_NEU_F400
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-04:00:00
#SBATCH --output=outputs/exp-6.2.1.2/<jobname>_%A-%a.out 
#SBATCH --account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

# bash scripts/exp-6.2.1.2/create_data.sh | tee -a ./outputs/exp-6.2.1.2/data.log
# bash scripts/exp-6.2.1.2/sft.sh | tee -a ./outputs/exp-6.2.1.2/sft.log
bash scripts/exp-6.2.1.2/eval_neutralization.sh | tee -a ./outputs/exp-6.2.1.2/eval.log