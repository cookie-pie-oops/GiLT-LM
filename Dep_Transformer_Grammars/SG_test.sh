#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=SG_mixing_4_psd_small_action_num_300_20.out

python SG_test_graphlayer.py \
    --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
    --beamsize 300 \
    --scorebeamsize 20 \