from typing import List
import torch
from dataclasses import dataclass
# from lightning.pytorch import seed_everything
import numpy as np
import numba
import numba.cuda as cuda
import argparse
import time
from helping_utils.logger import configure_logger, get_logger
from model_bllip_dep import TransformerGrammar, BiaffineAttention, calculate_depth, dijkstra
from beam_search_utils import update_beam, load_vocab, load_data, add_to_all
from copy import deepcopy
import json
import re
import os
from tqdm import tqdm
import sentencepiece as spm
import math
import heapq
from itertools import count
import random
import itertools

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models/graphlayer.pt', type=str)
parser.add_argument('--beamsize', default=100, type=int)
parser.add_argument('--scorebeamsize', default=20, type=int)

if __name__ == "__main__":
    # seed_everything(42)
    args = parser.parse_args()
    counter = count()
    configure_logger('logs/BLiMP_test.log')
    logger = get_logger()
    vocab_size, pad, bos, eos, startofword_id, vocab = load_vocab('../data_process/spm_parsing/BLLIP_spm.vocab')
    torch.manual_seed(123456)
    np.random.seed(123456)
    random.seed(12345)
    original = []
    subtoken_begins = []
    original_length = []
    original_seq_length = []

    model_path = args.model_path
    beamsize = args.beamsize
    scorebeamsize = args.scorebeamsize
    logger.info("Model path: {}".format(model_path))
    logger.info("Beam size: {}".format(beamsize))
    logger.info("Score beam size: {}".format(scorebeamsize))
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model = checkpoint['model']
    model.eval()
    model.to(device)
    biaffine_model = checkpoint['biaffine_model']
    biaffine_model.eval()
    biaffine_model.to(device)
    biaffine_model.set_temperature(1.0)
    
    sp = spm.SentencePieceProcessor(model_file='../data_process/spm_parsing/BLLIP_spm.model')
    file_list = os.listdir("/home/huangty/BLiMP_data/json/.")
    final_acc = []
    for file in file_list:
        with open("/home/huangty/BLiMP_data/json/" + file, 'r') as f:
            data = f.readlines()
            sample_list = [json.loads(ds.strip()) for ds in data]
        sample_num = (len(sample_list) - 1) // 10 + 1
        sample_index = [x * 10 for x in range(sample_num)]
        sample_list = [sample_list[i] for i in sample_index]
        # sample_list = random.sample(sample_list, int(len(sample_list) / 10))
        logger.info(file[:-6])
        acc = 0.0
        for idx in tqdm(range(len(sample_list))):
            examples = {"sentence_good": sample_list[idx]["sentence_good"], "sentence_bad": sample_list[idx]["sentence_bad"]}
            phen2surprisals = {}
            for phen in examples:
                assert examples[phen][-1] in ["!", ".", "?"]
                text = examples[phen][:-1] + " " + examples[phen][-1]
                encoded = sp.Encode(text, out_type=int)
                encoded.insert(0, 1)
                encoded.append(2)
                word_idx = -1
                prev_idx = -1
                start_predict_new_word = [1 if startofword_id[encoded[i + 1]] else 0 for i in range(len(encoded) - 1)]
                sent_index_to_id = []
                count_num = 0
                for word_id in encoded:
                    if word_id in [bos, eos]:
                        sent_index_to_id.append(-1)
                        continue
                    elif startofword_id[word_id] == 1:
                        count_num += 1
                    sent_index_to_id.append(count_num)
                
                scores, _ = update_beam(encoded, model, biaffine_model, start_predict_new_word, 
                    sent_index_to_id, beamsize, scorebeamsize, device, logger)
                
                # txl eval
                # _, prob = model([encoded], None, None, use_mask=None)
                # prob = prob.log_softmax(-2)
                # scores = [-prob[i, encoded[i + 1]].item() for i in range(len(encoded[1:]))]
                # scores.insert(0, 0)
                # scores = list(itertools.accumulate(scores))   

                phen2surprisals[phen] = scores[-2]
            
            if phen2surprisals["sentence_good"] < phen2surprisals["sentence_bad"]:
                acc += 1
        
        logger.info(f"correct rate: {acc / len(sample_list)}")
        final_acc.append(acc / len(sample_list))
        logger.info(f"mean correct rate up to now: {np.mean(final_acc)}")
    
    logger.info(f"final correct rate: {np.mean(final_acc)}")