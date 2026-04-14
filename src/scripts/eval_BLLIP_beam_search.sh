#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=GiLT_parse_pas_od.out
# ../data_process/RTE/RTE_TRAIN_token_parse.txt \
# ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \

# python BLLIP_beam_search.py \
#     --test_file ../data_process/ACE_arrow/psd_ood_test.txt \
#     --model_path models/GiLT_large_dm_4_1:0.2_3mixing_easydepth.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.ood.dm.json \
#     --parse

# python BLLIP_beam_search.py \
#     --test_file ../data_process/ACE_arrow/psd_id_test.txt \
#     --model_path models/GiLT_large_dm_4_1:0.2_3mixing_easydepth.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.id.dm.json \
#     --parse

python BLLIP_beam_search.py \
    --test_file ../data_process/ACE_arrow/psd_ood_test.txt \
    --model_path models/GiLT_large_psd_4_1:0.2_mixing_-pred.pt \
    --beamsize 300 \
    --scorebeamsize 5 \
    --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.ood.psd.json \
    --parse

# python BLLIP_beam_search.py \
#     --test_file ../data_process/ACE_arrow/psd_id_test.txt \
#     --model_path models/GiLT_large_psd_4_1:0.2_mixing_-pred.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.id.psd.json \
#     --parse

# python BLLIP_beam_search.py \
#     --test_file ../data_process/ACE_arrow/psd_ood_test.txt \
#     --model_path models/GiLT_large_pas_4_1:0.2_3mixing_easydepth.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.ood.pas.json \
#     --parse

# python BLLIP_beam_search.py \
#     --test_file ../data_process/ACE_arrow/psd_id_test.txt \
#     --model_path models/GiLT_large_pas_4_1:0.2_3mixing_easydepth.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --output_file /home/huangty/SDP2015_dataset/sdp2014_2015/data/2015/test/GiLT-output/en.id.pas.json \
#     --parse