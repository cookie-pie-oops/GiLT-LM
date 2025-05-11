#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=BLLIP_beamsearch_actionnum_100_20.out
# --test_file ../data_process/RTE/RTE_TRAIN_token_parse.txt \

python BLLIP_beam_search.py \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
    --beamsize 100 \
    --scorebeamsize 20 \
    --parse_file ../data_process/RTE/RTE_TRAIN_psd_multiarrow_parse.txt \
    --finetuneset rte
    # --parse