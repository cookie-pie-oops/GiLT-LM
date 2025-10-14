#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH -c 1
#SBATCH --mem=1M
#SBATCH -G 1
#SBATCH --output=GiLT_small_psd_4_3mixing_easydepth_1014v1.out

# --model_file models/graphlayer_$DATASIZE\_$DATASET\_4_1:$LOSSRATIO\_$RELTYPE\_DTG.pt \

export DATASET=psd
export DATASIZE=small
export RELTYPE=mixing
export MIXING_NUM=3

export MODEL_PATH=GiLT_$DATASIZE\_$DATASET\_4_3$RELTYPE\_easydepth_1014v1

python train_graphLayer.py \
    --train_file  ../data_process/token_level/BLLIP_LG_TRAIN_SPM_TOK.csv \
    --dev_file ../data_process/token_level/BLLIP_LG_DEV_SPM_TOK.csv \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --train_arrow_file ../data_process/ACE_arrow/TRAIN_$DATASET\_ACE_multiarrow.txt \
    --dev_arrow_file ../data_process/ACE_arrow/DEV_$DATASET\_ACE_multiarrow.txt \
    --test_arrow_file ../data_process/ACE_arrow/TEST_$DATASET\_ACE_multiarrow.txt \
    --log_file logs/log_$MODEL_PATH.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --save_path models/$MODEL_PATH.pt \
    --mixing_num $MIXING_NUM \
    --rel_type $RELTYPE \
    --dataset $DATASIZE \
    --biaffine_head 1 \
    --biaffine_out_dim 1024 \
    --loss_alpha 1.0 \
    --loss_beta 0.6 \
    --loss_gamma 0.2 \
    --attn_mask graphlayer \
    --sentence_level \
    --emb_lr_multiplier 1.0 \
    --return_h \
    --gpu 0 \
    --batch_size 64 \
    --w_dim 1024 \
    --n_head 8 \
    --d_head 128 \
    --d_inner 4096 \
    --transformer_lr_ratio 1 \
    --num_layers 16 \
    --max_relative_length 62 \
    --min_relative_length -1 \
    --seed 12345 \
    --weight_decay 0 \
    --max_grad_norm 3.0 \
    --num_epochs 4 \
    --decay_epochs 4 \
    --scheduler cosine \
    --optimizer adamw \
    --start_lr 1e-7 \
    --max_lr 1.5e-4 \
    --eta_min 3e-7 \
    --stable_lr 3e-7 \
    --lr_warm_step 8000 \
    --dropout 0.0 \
    --dropoutatt 0 \
    --eval_interval 500 \
    --eval_batch_size 8 \
    --log_every 20 \
    --min_lr 5e-7 \
    --decay_interval 8 \

python BLLIP_beam_search.py \
    --test_file ../data_process/token_level/BLLIP_LG_DEV_SPM_TOK.csv \
    --model_path models/$MODEL_PATH.pt \
    --beamsize 300 \
    --scorebeamsize 5 \
    --parse_file ../data_process/RTE/RTE_TRAIN_psd_multiarrow_parse.txt \
    --finetuneset rte

python SG_test_graphlayer.py \
    --model_path models/$MODEL_PATH.pt \
    --beamsize 300 \
    --scorebeamsize 5 \

# python BLiMP_graphlayer.py \
#     --model_path models/$MODEL_PATH.pt \
#     --beamsize 300 \
#     --scorebeamsize 5 \

# Hanlp
# --train_arrow_file ../data_process/transition_sequence/BLLIP_LG_TRAIN_$DATASET\_multiarrow.txt \
# --dev_arrow_file ../data_process/transition_sequence/BLLIP_LG_DEV_$DATASET\_multiarrow.txt \
# --test_arrow_file ../data_process/transition_sequence/BLLIP_LG_TEST_$DATASET\_multiarrow.txt \

# Supar
    # --train_arrow_file ../data_process/supar_arrow/TRAIN_$DATASET\_Supar_multiarrow.txt \
    # --dev_arrow_file ../data_process/supar_arrow/DEV_$DATASET\_Supar_multiarrow.txt \
    # --test_arrow_file ../data_process/supar_arrow/TEST_$DATASET\_Supar_multiarrow.txt \

# ACE
    # --train_arrow_file ../data_process/ACE_arrow/TRAIN_$DATASET\_ACE_multiarrow.txt \
    # --dev_arrow_file ../data_process/ACE_arrow/DEV_$DATASET\_ACE_multiarrow.txt \
    # --test_arrow_file ../data_process/ACE_arrow/TEST_$DATASET\_ACE_multiarrow.txt \

# DTG
    # --train_arrow_file ../data_process/DTG_data/dtg_train_multiarrow.txt \
    # --dev_arrow_file ../data_process/DTG_data/dtg_dev_multiarrow.txt \
    # --test_arrow_file ../data_process/DTG_data/dtg_test_multiarrow.txt \