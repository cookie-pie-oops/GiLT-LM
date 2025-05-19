#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=BLiMP_txl_full_2.out
# models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt
python BLiMP_graphlayer.py \
    --model_path models/large_tok_txl_seed_4.pt \
    --beamsize 300 \
    --scorebeamsize 20 \