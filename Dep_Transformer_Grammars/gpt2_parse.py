import argparse
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer, GPT2LMHeadModel #4.44.1
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
import csv
import copy
from tqdm import tqdm
from train_graphLayer import load_multiarrow, predicate_alignment
from model_bllip_dep import calculate_depth, dijkstra, BiaffineAttention
from helping_utils.logger import configure_logger, get_logger
from gpt2_posttraining import synchronize_arrows, GiLTGPT2LMHead, startofword_id
from beam_search_utils import GiLT_GPT2_update_beam, get_best_graph, get_action_from_graph
logger = get_logger()

def load_data(filename, batchsize=8, shuffle=True):

    with open(filename, 'r') as f:
        sents = [line.strip() for line in f.readlines()]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]

parser = argparse.ArgumentParser()
parser.add_argument('--parse_file_path', default='../data_process/MRPC/MRPC_TEST.txt', type=str)
parser.add_argument('--beamsize', default=20, type=int)
parser.add_argument('--scorebeamsize', default=5, type=int)
parser.add_argument('--save_arrow_path', default='../data_process/MRPC/MRPC_parse_test_multiarrow.txt', type=str)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained("/home/huangty/GPT2/medium355M")
    tokenizer.add_prefix_space = True
    model = GiLTGPT2LMHead.from_pretrained("/home/huangty/GPT2/medium355M")
    model.load_state_dict(torch.load("models/GiLT_gpt2_medium_post.pt", map_location=device))
    biaffine_model = BiaffineAttention(4096, 1024, type="Multi")
    biaffine_model.load_state_dict(torch.load("models/GiLT_gpt2_biaffine.pt", map_location=device))
    test_data = load_data(args.parse_file_path, batchsize=1, shuffle=False)
    configure_logger("logs/GiLT_gpt2_parse.log")

    beamsize = args.beamsize
    scorebeamsize = args.scorebeamsize
    logger.info(f"parse file path: {args.parse_file_path}")
    logger.info(f"beamsize: {args.beamsize}, scorebeamsize: {args.scorebeamsize}")
    logger.info(f"save arrow path: {args.save_arrow_path}")
    logger.info(f"load model from models/GiLT_gpt2_medium_post.pt")

    model.eval()
    biaffine_model.eval()
    model.to(device)
    biaffine_model.to(device)
    biaffine_model.set_temperature(1.0)

    bos_eos_id = 50256
    fw = open(args.save_arrow_path, 'w')
    for idx in tqdm(range(len(test_data))):
        text = test_data[idx][0]
        encoded = [bos_eos_id] + tokenizer.encode(tokenizer.decode(tokenizer.encode(text)).strip()) + [bos_eos_id]
        start_predict_new_word = [1 if startofword_id[encoded[i + 1]] else 0 for i in range(len(encoded) - 1)]
        sent_index_to_id = synchronize_arrows([encoded])[0]
        scores, beam_with_graph = GiLT_GPT2_update_beam(encoded, model, biaffine_model, start_predict_new_word,
            sent_index_to_id, beamsize, scorebeamsize, device, logger)
        best_graph = get_best_graph(beam_with_graph)
        arrow_dict = get_action_from_graph(best_graph)
        fw.write(json.dumps(arrow_dict, ensure_ascii=False)+'\n')