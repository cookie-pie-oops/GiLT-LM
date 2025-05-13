#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=parse_RTE_small_20_20.out
# ../data_process/RTE/RTE_TRAIN_token_parse.txt \
# ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \

python BLLIP_beam_search.py \
    --test_file ../data_process/RTE/RTE_TRAIN_token_parse.txt \
    --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
    --beamsize 5 \
    --scorebeamsize 5 \
    --parse_file ../data_process/RTE/RTE_TRAIN_psd_multiarrow_parse.txt \
    --finetuneset rte \
    --parse

# python BLLIP_beam_search.py \
#     --test_file ../data_process/RTE/RTE_TEST_token_parse.txt \
#     --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
#     --beamsize 20 \
#     --scorebeamsize 20 \
#     --parse_file ../data_process/RTE/RTE_TEST_psd_multiarrow_parse.txt \
#     --finetuneset rte \
#     --parse

python BLLIP_beam_search.py \
    --test_file ../data_process/RTE/RTE_DEV_token_parse.txt \
    --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
    --beamsize 5 \
    --scorebeamsize 5 \
    --parse_file ../data_process/RTE/RTE_DEV_psd_multiarrow_parse.txt \
    --finetuneset rte \
    --parse