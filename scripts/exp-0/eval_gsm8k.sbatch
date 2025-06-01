#!/bin/bash
#SBATCH --job-name=Eval_70B_GSM8K
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:4
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-3:00:00
#SBATCH --output=outputs/exp-0/<jobname>_%A-%a.out 
#SBATCH --account=pi_mccallum_umass_edu

bash scripts/exp-0/eval_gsm8k.sh