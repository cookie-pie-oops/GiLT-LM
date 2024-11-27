#!/bin/bash
export DATASET=psd
# --model_file models/trans_psd_txl_spm_4.pt \
python train_sdp.py \
    --train_file  ../data_process/transition_sequence_txl/BLLIP_LG_TRAIN_$DATASET.csv \
    --dev_file ../data_process/transition_sequence_txl/BLLIP_LG_DEV_$DATASET.csv \
    --test_file ../data_process/transition_sequence_txl/BLLIP_LG_TEST_$DATASET.csv \
    --train_arrow_file ../data_process/transition_sequence/BLLIP_LG_TRAIN_$DATASET\_arrow.txt \
    --dev_arrow_file ../data_process/transition_sequence/BLLIP_LG_DEV_$DATASET\_arrow.txt \
    --test_arrow_file ../data_process/transition_sequence/BLLIP_LG_TEST_$DATASET\_arrow.txt \
    --log_file logs/log_txl_pointer_$DATASET\_4_1:10.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --save_path models/txl_pointer_$DATASET\_4_1:10.pt \
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
    --TBloss_ratio 10 \
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
    --lr_warm_step 8000 \
    --dropout 0.0 \
    --dropoutatt 0 \
    --eval_interval 500 \
    --eval_batch_size 8 \
    --log_every 20 \
    --min_lr 5e-7 \
    --decay_interval 8 \

