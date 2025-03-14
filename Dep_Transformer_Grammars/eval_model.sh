#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=graphlayer_small_psd_4_1:1_degree_ACE_sampling_300.out
export DATASET=psd
export RELTYPE=degree
export EVALTYPE=estimate    #estimate

 # don't need sampling dev

python eval_model.py \
    --dev_file ../data_process/token_level/BLLIP_LG_DEV_SPM_TOK.csv \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --dev_arrow_file ../data_process/ACE_arrow/DEV_$DATASET\_ACE_multiarrow.txt \
    --test_arrow_file ../data_process/ACE_arrow/TEST_$DATASET\_ACE_multiarrow_300.txt \
    --log_file logs/eval.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/graphlayer_small_$DATASET\_4_1:1_$RELTYPE\_ACE.pt \
    --eval_type $EVALTYPE \
    --sampling_num 300 \
    --eval_batch_size 300 \
    --rel_type $RELTYPE \
    --sentence_level \
    --emb_lr_multiplier 2.0 \
    --attn_mask graphlayer \
    --gpu 0 \
    --batch_size 64 \
    --w_dim 1024 \
    --n_head 8 \
    --d_head 128 \
    --d_inner 4096 \
    --num_layers 16 \
    --max_relative_length 62 \
    --min_relative_length -1 \
    --seed 12345 \
    --weight_decay 0 \
    --max_grad_norm 3.0 \

#Supar
    # --dev_arrow_file ../data_process/supar_arrow/DEV_$DATASET\_Supar_multiarrow.txt \
    # --test_arrow_file ../data_process/supar_arrow/TEST_$DATASET\_Supar_multiarrow.txt \
# Hanlp
    # --dev_arrow_file ../data_process/transition_sequence/BLLIP_LG_DEV_$DATASET\_multiarrow.txt \
    # --test_arrow_file ../data_process/transition_sequence/BLLIP_LG_TEST_$DATASET\_multiarrow.txt \
# ACE
    # --dev_arrow_file ../data_process/ACE_arrow/DEV_$DATASET\_ACE_multiarrow.txt \
    # --test_arrow_file ../data_process/ACE_arrow/TEST_$DATASET\_ACE_multiarrow.txt \
# DTG
    # --test_arrow_file ../data_process/DTG_data/dtg_test_multiarrow.txt \