#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=BLiMP_actionnum_small_100_20.out

python BLiMP_graphlayer.py \
    --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
    --beamsize 100 \
    --scorebeamsize 20 \