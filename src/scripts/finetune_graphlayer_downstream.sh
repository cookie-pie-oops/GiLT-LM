#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH -G 1
#SBATCH --output=finetune_psd_sst2_4layers_pretrainembedding.out

export fix_lr=3e-6
export epoch=5
export dataset=SST2
export finetune_set=sst2
export eval_interval=100
export DATASIZE=large
export RELTYPE=mixing
export LOSSRATIO=0.2
export PARSE=psd
# baseLM
python train_graphLayer.py \
    --train_file  ../data_process/$dataset/$dataset\_TRAIN_token.txt \
    --dev_file ../data_process/$dataset/$dataset\_DEV_token.txt \
    --test_file ../data_process/$dataset/$dataset\_TEST_token.txt \
    --train_arrow_file ../data_process/$dataset/$dataset\_TRAIN_$PARSE\_multiarrow_parse.txt \
    --dev_arrow_file ../data_process/$dataset/$dataset\_DEV_$PARSE\_multiarrow_parse.txt \
    --test_arrow_file ../data_process/$dataset/$dataset\_TEST_$PARSE\_multiarrow_parse.txt \
    --write_test_output ../data_process/$dataset/$dataset\_TEST_pred_$PARSE.tsv \
    --sts_train_path ../data_process/STS/STS_TRAIN_score.txt \
    --sts_dev_path ../data_process/STS/STS_DEV_score.txt \
    --sts_test_path ../data_process/STS/STS_TEST_score.txt \
    --log_file logs/finetune_$PARSE\_graphlayer_$finetune_set.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/graphlayer_large_4_psd_1_0.2_mixing_predict_ahead_graph_rel_split_actionnum_head.pt \
    --save_path models/finetune_$PARSE\_graphlayer_$finetune_set\_4layers_pretrainembedding.pt \
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
    --seed 123456 \
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
    --eval_batch_size 8 \
    --log_every 10 \
    --min_lr $fix_lr \
    --decay_interval 8 \

