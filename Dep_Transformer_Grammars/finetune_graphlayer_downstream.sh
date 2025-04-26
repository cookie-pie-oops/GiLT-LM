#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --output=finetune_psd_MRPC.out

# 对于parse完的，将format的sent_idx_to_id设置为-1即可
# eval interval: sst2 50, mrpc 10, rte 20
export fix_lr=3e-6  # sst2 1e-5, 3e-6
export epoch=15  # sst2 5, mrpc 15, rte 5
export dataset=MRPC
export finetune_set=mrpc
export eval_interval=20
export DATASIZE=large
export RELTYPE=mixing
export LOSSRATIO=0.8
# baseLM
python train_graphLayer.py \
    --train_file  ../data_process/$dataset/$dataset\_TRAIN_token.txt \
    --dev_file ../data_process/$dataset/$dataset\_DEV_token.txt \
    --test_file ../data_process/$dataset/$dataset\_TEST_token.txt \
    --train_arrow_file ../data_process/$dataset/$dataset\_TRAIN_psd_multiarrow_2.txt \
    --dev_arrow_file ../data_process/$dataset/$dataset\_DEV_psd_multiarrow_2.txt \
    --test_arrow_file ../data_process/$dataset/$dataset\_TEST_psd_multiarrow_2.txt \
    --log_file logs/finetune_psd_graphlayer_$finetune_set.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/graphlayer_small_psd_4_1:0.8_mixing_ACE_predict_ahead_graph_rel.pt \
    --save_path models/finetune_psd_graphlayer_$finetune_set.pt \
    --rel_type $RELTYPE \
    --BTloss_ratio $LOSSRATIO \
    --dataset $DATASIZE \
    --degree_len 400 \
    --distance_len 400 \
    --depth_len 150 \
    --predepth_len 74 \
    --attn_mask graphlayer \
    --return_h \
    --finetune $finetune_set \
    --sentence_level \
    --emb_lr_multiplier 1.0 \
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
    --num_epochs $epoch \
    --decay_epochs $epoch \
    --scheduler cosine \
    --optimizer adamw \
    --start_lr $fix_lr \
    --max_lr $fix_lr \
    --eta_min $fix_lr \
    --lr_warm_step 8000 \
    --dropout 0.0 \
    --dropoutatt 0 \
    --eval_interval $eval_interval \
    --eval_batch_size 32 \
    --log_every 10 \
    --min_lr $fix_lr \
    --decay_interval 8 \

