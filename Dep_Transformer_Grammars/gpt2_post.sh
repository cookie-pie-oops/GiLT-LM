#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=GiLT_gpt2_RTE.out

python gpt2_posttraining.py

export TASK=RTE
# python gpt2_finetune.py