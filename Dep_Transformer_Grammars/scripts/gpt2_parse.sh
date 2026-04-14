#!/bin/bash
#SBATCH -t 10-00:00:00
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --output=./outputs/GiLT_GPT2_parse_8.out

python gpt2_parse.py \
    --parse_file_path ../data_process/STS/STS_DEV.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/STS/STS_parse_dev_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/RTE/RTE_DEV.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/RTE/RTE_parse_dev_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/STS/STS_TEST.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/STS/STS_parse_test_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/RTE/RTE_TEST.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/RTE/RTE_parse_test_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/RTE/RTE_TRAIN.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/RTE/RTE_parse_train_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/STS/STS_TRAIN.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/STS/STS_parse_train_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/SST2/SST2_DEV.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/SST2/SST2_parse_dev_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/MRPC/MRPC_DEV.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/MRPC/MRPC_parse_dev_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/SST2/SST2_TEST.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/SST2/SST2_parse_test_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/MRPC/MRPC_TEST.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/MRPC/MRPC_parse_test_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/MRPC/MRPC_TRAIN.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/MRPC/MRPC_parse_train_multiarrow.txt

python gpt2_parse.py \
    --parse_file_path ../data_process/SST2/SST2_TRAIN.txt \
    --beamsize 20 \
    --scorebeamsize 5 \
    --save_arrow_path ../data_process/SST2/SST2_parse_train_multiarrow.txt

# python gpt2_parse.py \
#     --parse_file_path ../data_process/GPT2-tokenizer/BLLIP_LG_TEST.txt \
#     --beamsize 300 \
#     --scorebeamsize 5 \
#     --save_arrow_path ../data_process/RTE/RTE_parse_train_multiarrow.txt