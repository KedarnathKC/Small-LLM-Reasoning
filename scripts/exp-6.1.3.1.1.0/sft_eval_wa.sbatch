#!/bin/bash
#SBATCH --job-name=SFT_8B_WA
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-04:00:00
#SBATCH	--account pi_mccallum_umass_edu
#SBATCH --output=outputs/exp-6.1.3.1.1.0/<jobname>_%A-%a.out 

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-6.1.3.1.1.0/create_data.sh | tee -a ./outputs/exp-6.1.3.1.1.0/data.log
bash scripts/exp-6.1.3.1.1.0/sft.sh | tee -a ./outputs/exp-6.1.3.1.1.0/sft.log
bash scripts/exp-6.1.3.1.1.0/eval_wa.sh | tee -a ./outputs/exp-6.1.3.1.1.0/eval.log