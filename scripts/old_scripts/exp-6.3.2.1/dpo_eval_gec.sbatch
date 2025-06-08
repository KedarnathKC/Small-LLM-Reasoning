#!/bin/bash
#SBATCH --job-name=M3_3B_GEC_F100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-12:00:00
#SBATCH --output=outputs/exp-6.3.2.1/<jobname>_%A-%a.out 
#SBATCH --account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-6.3.2.1/create_data.sh | tee -a ./outputs/exp-6.3.2.1/data.log
bash scripts/exp-6.3.2.1/dpo.sh | tee -a ./outputs/exp-6.3.2.1/dpo.log
bash scripts/exp-6.3.2.1/eval_gec.sh | tee -a ./outputs/exp-6.3.2.1/eval.log