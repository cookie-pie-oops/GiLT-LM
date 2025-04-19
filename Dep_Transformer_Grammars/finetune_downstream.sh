#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=finetune_txl_sst2.out

# 对于parse完的，将format的sent_idx_to_id设置为-1即可

# baseLM
python train_dep.py \
    --train_file  ../data_process/SST-2/SST2_TRAIN_token.txt \
    --dev_file ../data_process/SST-2/SST2_DEV_token.txt \
    --test_file ../data_process/SST-2/SST2_TEST_token.txt \
    --log_file logs/finetune_txl_sst2.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/large_tok_txl_seed_4.pt \
    --save_path models/finetune_txl_sst2.pt \
    --finetune sst2 \
    --sentence_level \
    --emb_lr_multiplier 1.0 \
    --attn_mask None \
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
    --num_epochs 10 \
    --decay_epochs 10 \
    --scheduler cosine \
    --optimizer adamw \
    --start_lr 7.5e-6 \
    --max_lr 7.5e-6 \
    --eta_min 7.5e-6 \
    --lr_warm_step 8000 \
    --dropout 0.0 \
    --dropoutatt 0 \
    --eval_interval 200 \
    --eval_batch_size 8 \
    --log_every 20 \
    --min_lr 7.5e-6 \
    --decay_interval 8 \

