#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=GiLT_gpt2_parse_RTE.out

# python gpt2_parse.py \
#     --parse_file_path ../data_process/STS/STS_DEV.txt \
#     --beamsize 20 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/STS/STS_parse_dev_multiarrow.txt

# python gpt2_parse.py \
#     --parse_file_path ../data_process/RTE/RTE_DEV.txt \
#     --beamsize 20 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/RTE/RTE_parse_dev_multiarrow.txt

# python gpt2_parse.py \
#     --parse_file_path ../data_process/STS/STS_TEST.txt \
#     --beamsize 20 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/STS/STS_parse_test_multiarrow.txt

# python gpt2_parse.py \
#     --parse_file_path ../data_process/RTE/RTE_TEST.txt \
#     --beamsize 20 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/RTE/RTE_parse_test_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/RTE/RTE_TRAIN.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/RTE/RTE_parse_train_multiarrow.txt

# python gpt2_parse.py \
#     --parse_file_path ../data_process/STS/STS_TRAIN.txt \
#     --beamsize 20 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/STS/STS_parse_train_multiarrow.txt
