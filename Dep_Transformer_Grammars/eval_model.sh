#!/bin/bash
python eval_model.py \
    --dev_file ../data_process/arc_with_point/BLLIP_LG_DEV_SPM_ARC_pas.txt \
    --test_file ../data_process/arc_with_point/BLLIP_LG_TEST_SPM_ARC_pas.txt \
    --log_file logs/eval.txt \
    --vocab_file ../data_process/BLLIP_spm.vocab \
    --model_file models/standard_arc_pas_txl_seed_4.pt \
    --sentence_level \
    --emb_lr_multiplier 2.0 \
    --attn_mask txl_arc \
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

