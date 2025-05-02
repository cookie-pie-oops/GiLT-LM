#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=BLiMP_mixing_4_psd_large_100_20.out

python BLiMP_graphlayer.py