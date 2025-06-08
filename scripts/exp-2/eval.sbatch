#!/bin/bash
#SBATCH --job-name=Eval_Neu_70B
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:4
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-01:00:00
#SBATCH --output=outputs/exp-2/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-2/eval.sh | tee -a ./outputs/exp-2/eval.log
