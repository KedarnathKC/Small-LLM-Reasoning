#!/bin/bash
#SBATCH --job-name=stu_logprobs_8B
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH -t 0-6:00:00
#SBATCH --output=outputs/exp-5.2.2.2/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

# Exit immediately if a command exits with a non-zero status
set -e

bash scripts/exp-5.2.2.2/studentlogprobs.sh | tee ./outputs/exp-5.2.2.2/studentlogprobs.log
bash scripts/exp-5.2.2.2/teacherlogprobs.sh | tee ./outputs/exp-5.2.2.2/teacherlogprobs.log