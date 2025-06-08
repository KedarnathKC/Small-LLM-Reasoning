#!/bin/bash
#SBATCH --job-name=sft_3B
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-4/<jobname>_%A-%a.out 
#SBATCH --account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-4/sft.sh | tee -a ./outputs/exp-4/sft.log
bash scripts/exp-4/eval.sh | tee -a ./outputs/exp-4/eval.log