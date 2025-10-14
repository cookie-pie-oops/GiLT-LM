#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=GiLT_50_10.out
# ../data_process/RTE/RTE_TRAIN_token_parse.txt \
# ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \

python BLLIP_beam_search.py \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --model_path models/GiLT_large_psd_4_1:0.2_mixing_-pred.pt \
    --beamsize 300 \
    --scorebeamsize 5 \
    --parse_file ../data_process/RTE/RTE_TRAIN_psd_multiarrow_parse.txt \
    --finetuneset rte

# python -m memory_profiler BLLIP_beam_search.py \
#     --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
#     --model_path models/GiLT_small_psd_4_1:0.2_3mixing_easydepth_fast_biaffine_AugmentedQ_oriarc.pt \
#     --beamsize 300 \
#     --scorebeamsize 20 \
#     --parse_file ../data_process/RTE/RTE_TEST_psd_multiarrow_parse.txt \
#     --finetuneset rte

# python BLLIP_beam_search.py \
#     --test_file ../data_process/RTE/RTE_DEV_token_parse.txt \
#     --model_path models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt \
#     --beamsize 5 \
#     --scorebeamsize 5 \
#     --parse_file ../data_process/RTE/RTE_DEV_psd_multiarrow_parse.txt \
#     --finetuneset rte \
#     --parse