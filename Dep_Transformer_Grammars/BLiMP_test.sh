#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH --output=BLLIP_PPL_3mixing_4_pas_large_action_num_300_20_easydepth_9add15_600out_SwiGLU.out
# models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_actionnum_head.pt
    # --model_path models/GiLT_large_psd_4_1:0.2_3mixing_easydepth_normaldistance.pt \
# --output=BLiMP_ppl_3mixing_4_psd_large_action_num_300_20_easydepth_normaldistance.out
# python BLiMP_graphlayer.py \
#     --model_path models/GiLT_large_psd_4_1:0.2_3mixing_easydepth_fp16.pt \
#     --beamsize 300 \
#     --scorebeamsize 20 \

# GiLT_small_psd_4_1:0.2_3mixing_easydepth_4096inner_fp16.pt
python BLLIP_beam_search.py \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --model_path models/GiLT_small_psd_4_1:0.2_3mixing_easydepth_9add15_600out_SwiGLU.pt \
    --beamsize 300 \
    --scorebeamsize 20 \
    --parse_file ../data_process/RTE/RTE_TRAIN_psd_multiarrow_parse.txt \
    --finetuneset rte

# python SG_test_graphlayer.py \
#     --model_path models/GiLT_large_psd_4_1:0.2_3mixing_easydepth_fp16.pt \
#     --beamsize 300 \
#     --scorebeamsize 20 \