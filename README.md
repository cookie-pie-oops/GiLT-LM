# Pushdown Layers Baseline
This branch is for the implementation of [Pushdown Layers: Encoding Recursive Structure in Transformer Language Models](https://arxiv.org/abs/2310.19089) on the same codebase as the main branch.

## Data Processing
We use the same dataset in this baseline as in the main branch: [BLLIP 1987-89 WSJ Corpus Release 1](https://catalog.ldc.upenn.edu/LDC2000T43).

First, you should ensure the dataset is downloaded and unzipped. The dataset should be in the directory `./Dataset/`,
e.g. `./Dataset/bliip_87_89_wsj/1987` (`.` is the working directory). Then, 
```bash
cd data_process
python BLLIP_process_con.py
```

This will generate the processed data in the directory `./data/processed_data/`, which contains 3 Arrow dataset splits (train/test/dev).

## Training
To train the model, you should first adjust parameters in `config.py`.
The configs in `config.py` has been set as the same as the main branch.
Use the following command:
```bash
. train_bllip_con.sh
```
This will start the training process. The training process will save the model checkpoints in the directory `./ckpt/`.

