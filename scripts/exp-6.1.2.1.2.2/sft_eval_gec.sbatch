#!/bin/bash
#SBATCH --job-name=FFT_3B_GEC_F400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-04:00:00
#SBATCH --output=outputs/exp-6.1.2.1.2.2/<jobname>_%A-%a.out 

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-6.1.2.1.2.2/create_data.sh | tee -a ./outputs/exp-6.1.2.1.2.2/data.log
bash scripts/exp-6.1.2.1.2.2/sft.sh | tee -a ./outputs/exp-6.1.2.1.2.2/sft.log
bash scripts/exp-6.1.2.1.2.2/eval_gec.sh | tee -a ./outputs/exp-6.1.2.1.2.2/eval.log