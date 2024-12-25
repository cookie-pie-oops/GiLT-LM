#!/bin/bash
export DATASET=psd
python eval_model.py \
    --dev_file ../data_process/token_level/BLLIP_LG_DEV_SPM_TOK.csv \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --dev_arrow_file ../data_process/transition_sequence/BLLIP_LG_DEV_$DATASET\_multiarrow.txt \
    --test_arrow_file ../data_process/transition_sequence/BLLIP_LG_TEST_$DATASET\_multiarrow.txt \
    --log_file logs/eval.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/graphlayer_small_$DATASET\_4_100:1_WkD_zero.pt \
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
    --eval_batch_size 8 \

