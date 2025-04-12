#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=SG_mixing_4_DP_small_100_10.out

python SG_test_graphlayer.py