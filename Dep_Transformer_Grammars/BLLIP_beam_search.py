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
from beam_search_utils import update_beam, load_vocab, load_data, add_to_all, get_best_graph, get_action_from_graph, add_format
import json
import re
import os
from tqdm import tqdm
import sentencepiece as spm
import math
import heapq
from itertools import count
import random

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
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model = checkpoint['model']
    biaffine_model = checkpoint['biaffine_model']
    model.eval()
    biaffine_model.eval()
    model.to(device)
    biaffine_model.to(device)

    biaffine_model.set_temperature(1.0)
    
    total_ppl = 0
    total_len = 0
    for idx in tqdm(range(len(test_data))):
        encoded = test_data[idx][0]
        start_predict_new_word = [1 if startofword_id[encoded[i + 1]] else 0 for i in range(len(encoded) - 1)]
        sent_index_to_id = test_index_to_id[idx][0]
        scores, beam_with_graph = update_beam(encoded, model, biaffine_model, start_predict_new_word,
            sent_index_to_id, beamsize, scorebeamsize, device, logger)
        if args.parse:
            best_graph = get_best_graph(beam_with_graph)
            arrow_dict = get_action_from_graph(best_graph)
            with open(args.parse_file, 'a') as f:
                f.write(json.dumps(arrow_dict, ensure_ascii=False)+'\n')
        total_ppl += scores[-1]
        total_len += len(encoded) - 1
        logger.info(f"sentence {idx + 1} ppl: {np.exp(scores[-1] / (len(encoded) - 1))}")
    
    logger.info(f"test ppl: {np.exp(total_ppl / total_len)}")