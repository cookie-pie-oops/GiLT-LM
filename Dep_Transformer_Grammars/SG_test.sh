#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=SG_3mixing_4_psd_large_action_num_300_20_easydepth_-depth.out

python SG_test_graphlayer.py \
    --model_path models/GiLT_large_psd_4_1:0.2_3mixing_easydepth_-depth.pt \
    --beamsize 300 \
    --scorebeamsize 20 \