#!/bin/bash
#SBATCH --job-name=DPO_3B
#SBATCH --partition=gpu-preempt,superpod-a100,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-02:00:00
#SBATCH --output=outputs/exp-4.3.1/<jobname>_%A-%a.out 
#SBATCH	--account pi_mccallum_umass_edu

bash scripts/exp-4.3.1/create_data.sh | tee ./outputs/exp-4.3.1/data.log
bash scripts/exp-4.3.1/dpo.sh | tee ./outputs/exp-4.3.1/dpo.log
bash scripts/exp-4.3.1/eval_gsm8k.sh | tee ./outputs/exp-4.3.1/eval.log