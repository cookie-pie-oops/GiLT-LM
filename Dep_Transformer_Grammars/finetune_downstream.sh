#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH -G 1
#SBATCH --output=finetune_txl_STS_1.out

# eval interval: sst2 50, mrpc 10, rte 20
export fix_lr=3e-6  # sst2 1e-5, 3e-6
export epoch=45  # sst2 5, mrpc 15, rte 5
export dataset=STS
export finetune_set=sts
export eval_interval=100
export random_seed=1234   # 1234, 12345, 123456
# baseLM
python train_dep.py \
    --train_file  ../data_process/$dataset/$dataset\_TRAIN_token.txt \
    --dev_file ../data_process/$dataset/$dataset\_DEV_token.txt \
    --test_file ../data_process/$dataset/$dataset\_TEST_token.txt \
    --log_file logs/finetune_txl_$finetune_set\_$random_seed.txt \
    --vocab_file ../data_process/spm_parsing/BLLIP_spm.vocab \
    --model_file models/large_tok_txl_seed_4.pt \
    --save_path models/finetune_txl_$finetune_set\_$random_seed.pt \
    --sts_train_path ../data_process/STS/STS_TRAIN_score.txt \
    --sts_dev_path ../data_process/STS/STS_DEV_score.txt \
    --sts_test_path ../data_process/STS/STS_TEST_score.txt \
    --write_test_output ../data_process/$dataset/$dataset\_TEST_pred.tsv \
    --finetune $finetune_set \
    --sentence_level \
    --emb_lr_multiplier 1.0 \
    --attn_mask None \
    --gpu 0 \
    --batch_size 128 \
    --w_dim 1024 \
    --n_head 8 \
    --d_head 128 \
    --d_inner 4096 \
    --num_layers 16 \
    --max_relative_length 62 \
    --min_relative_length -1 \
    --seed $random_seed \
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

