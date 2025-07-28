#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=DTG_30_10.out

python beam_search_standard.py