#!/bin/bash
#SBATCH --job-name=tr_logprobs
#SBATCH --partition=superpod-a100
#SBATCH --account pi_mccallum_umass_edu
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH -t 0-2:00:00
#SBATCH --output=outputs/exp-2.1.1/<jobname>_%A-%a.out 

bash scripts/exp-2.1.1/teacherlogprobs.sh 