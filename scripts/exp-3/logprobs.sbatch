#!/bin/bash
#SBATCH --job-name=stu_logprobs_3B
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH -t 0-1:00:00
#SBATCH --output=outputs/exp-3/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-3/studentlogprobs.sh | tee -a ./outputs/exp-3/studentlogprobs.log
bash scripts/exp-3/teacherlogprobs.sh | tee -a ./outputs/exp-3/teacherlogprobs.log