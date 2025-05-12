#!/bin/bash

sbatch scripts/exp-6.1.1.1/dpo_eval_neutralization.sbatch
sbatch scripts/exp-6.1.1.2/dpo_eval_neutralization.sbatch
sbatch scripts/exp-6.1.1.3/dpo_eval_neutralization.sbatch

sbatch scripts/exp-6.1.2.1/dpo_eval_gec.sbatch
sbatch scripts/exp-6.1.2.2/dpo_eval_gec.sbatch
sbatch scripts/exp-6.1.2.3/dpo_eval_gec.sbatch