#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=GiLT_gpt2_post.out

python gpt2_posttraining.py

# python gpt2_finetune.py