#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=./outputs/GiLT_gpt2_finetune_SST2_8.out

python gpt2_posttraining.py

# export TASK=SST2
# python gpt2_finetune.py