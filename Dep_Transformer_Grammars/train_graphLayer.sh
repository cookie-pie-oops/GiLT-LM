#!/bin/bash
export DATASET=psd
# --model_file models/graphlayer_small_$DATASET\_4_100:1_10Do+Di.pt \
python train_graphLayer.py \
    --train_file  ../data_process/token_level/BLLIP_LG_TRAIN_SPM_TOK.csv \
    --dev_file ../data_process/token_level/BLLIP_LG_DEV_SPM_TOK.csv \
    --test_file ../data_process/token_level/BLLIP_LG_TEST_SPM_TOK.csv \
    --train_arrow_file ../data_process/transition_sequence/BLLIP_LG_TRAIN_$DATASET\_multiarrow.txt \
    --dev_arrow_file ../data_process/transition_sequence/BLLIP_LG_DEV_$DATASET\_multiarrow.txt \
    --test_arrow_file ../data_process/transition_sequence/BLLIP_LG_TEST_$DATASET\_multiarrow.txt \
    --log_file logs/log_graphlayer_large_$DATASET\_4_100:1_10Do+Di.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --save_path models/graphlayer_large_$DATASET\_4_100:1_10Do+Di.pt \
    --dataset large \
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
    --proj_dim 1024 \
    --transformer_lr_ratio 1 \
    --TBloss_ratio 0.01 \
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

