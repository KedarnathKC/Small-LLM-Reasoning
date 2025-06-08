#!/bin/bash
#SBATCH --job-name=Eval_WA_70B
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH -t 0-06:00:00
#SBATCH --output=outputs/exp-5.1.1/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-5.1.1/eval_wa.sh | tee -a ./outputs/exp-5.1.1/eval.log
