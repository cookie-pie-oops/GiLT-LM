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
from copy import deepcopy
from beam_search_utils import update_beam, load_vocab, load_data, add_to_all, get_best_graph, get_action_from_graph, add_format, update_concat
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

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', default='../data_process//test_bllip_action.csv', type=str)
parser.add_argument('--model_path', default='models/graphlayer.pt', type=str)
parser.add_argument('--beamsize', default=100, type=int)
parser.add_argument('--scorebeamsize', default=20, type=int)
parser.add_argument('--parse', default=False, action='store_true')
parser.add_argument('--parse_file', default='../data_process/test_psd_multiarrow_parsing.txt', type=str)
parser.add_argument('--finetuneset', default=None, type=str)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if __name__ == "__main__":
    # seed_everything(42)
    args = parser.parse_args()
    configure_logger('logs/BLLIP_ppl.log')
    logger = get_logger()
    vocab_size, pad, bos, eos, startofword_id, vocab = load_vocab('../data_process/spm_parsing/BLLIP_spm.vocab')
    test_data = load_data(args.test_file, batchsize=1, shuffle=False)
    test_data, startofword_test, test_length, test_index_to_id = add_to_all(test_data, vocab_size, pad, bos, eos, startofword_id)
    # if args.finetuneset is not None:
    #     test_index_to_id = add_format(test_data, test_index_to_id, args.finetuneset)
    torch.manual_seed(123456)
    np.random.seed(123456)

    model_path = args.model_path
    beamsize = args.beamsize
    scorebeamsize = args.scorebeamsize
    logger.info("Model path: {}".format(model_path))
    logger.info("Beam size: {}".format(beamsize))
    logger.info("Score beam size: {}".format(scorebeamsize))
    logger.info(f"parse: {args.parse}")
    if args.parse:
        logger.info(f"parse file: {args.parse_file}")
        fw = open(args.parse_file, 'w')
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model = checkpoint['model']
    model.eval()
    model.to(device)
    biaffine_model = checkpoint['biaffine_model']
    biaffine_model.eval()
    biaffine_model.to(device)
    biaffine_model.set_temperature(1.0)
    
    total_ppl = 0
    total_len = 0
    start_time = time.time()
    for idx in tqdm(range(len(test_data))): #[:8]
        encoded = test_data[idx][0]
        # if idx <= 2162:
        #     total_len += len(encoded) - 1
        #     total_ppl = np.log(14.521231167495714) * total_len
        #     continue
        start_predict_new_word = [1 if startofword_id[encoded[i + 1]] else 0 for i in range(len(encoded) - 1)]
        sent_index_to_id = test_index_to_id[idx][0]
        scores, beam_with_graph = update_beam(encoded, model, biaffine_model, start_predict_new_word,
            sent_index_to_id, beamsize, scorebeamsize, device, logger)
        
        # txl eval
        # _, prob = model([encoded], None, None, use_mask=None)
        # prob = prob.log_softmax(-2)
        # scores = [-prob[i, encoded[i + 1]].item() for i in range(len(encoded[1:]))]
        # scores.insert(0, 0)
        # scores = list(itertools.accumulate(scores))

        # txl infer
        # cache_k, cache_v = None, None
        # scores = [0.0]
        # temp_score = 0.0
        # tokens = torch.LongTensor(encoded[:-1]).to(device).reshape(1, -1)
        # for i in range(tokens.shape[1]):
        #     batch = tokens.shape[0]
        #     seq_len = i + 1
        #     prob, new_k, new_v = model.TXL_inference(tokens[:, i].repeat(batch, 1), cache_k, cache_v, seq_len)
        #     cache_k = update_concat(cache_k, new_k[0:1, :, :])
        #     cache_v = update_concat(cache_v, new_v[0:1, :, :])
        #     temp_score -= prob[0, 0, encoded[i+1]].item()
        #     scores.append(temp_score)

        if args.parse:
            best_graph = get_best_graph(beam_with_graph)
            arrow_dict = get_action_from_graph(best_graph)
            fw.write(json.dumps(arrow_dict, ensure_ascii=False)+'\n')
        total_ppl += scores[-1]
        total_len += len(encoded) - 1
        logger.info(f"sentence {idx + 1} ppl: {np.exp(scores[-1] / (len(encoded) - 1))}")
        logger.info(f"test ppl: {np.exp(total_ppl / total_len)}")
    end_time = time.time()
    logger.info(f"test ppl: {np.exp(total_ppl / total_len)}")
    logger.info(f"tokens per second: {total_len / (end_time - start_time)} tokens/s")
    logger.info(f"Max memory usage: {round((torch.cuda.max_memory_allocated()) / 1024 / 1024 / 1024, 2)} GB")