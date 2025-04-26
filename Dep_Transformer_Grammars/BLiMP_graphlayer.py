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
import json
import re
import os
from tqdm import tqdm
import sentencepiece as spm
import math
import heapq
from itertools import count
import random


def load_vocab(path):
    vocab_file = path

    pad_id = None
    bos_id = None
    eos_id = None

    with open(vocab_file, 'r') as f:
        vocab = [line.strip().split()[0] for line in f.readlines()]
        vocab_size = len(vocab)
        startofword_id = [0 for _ in range(vocab_size)]

        for i in range(0, len(vocab)):
            if vocab[i] == '<pad>':
                pad_id = i
            elif vocab[i] == '<s>':
                bos_id = i
            elif vocab[i] == '</s>':
                eos_id = i
            elif vocab[i].startswith('▁'):
                startofword_id[i] = 1

    return vocab_size, pad_id, bos_id, eos_id, startofword_id, vocab


def predicate_alignment(hidden, sents_index_to_id, word_emb):    # for predicates
    batch_words_input = []
    for sent_hidden, sent_index_to_id, sent_word_emb in zip(hidden, sents_index_to_id, word_emb):
        words_input = []
        for j in range(len(sent_index_to_id) - 1):
            # if sent_index_to_id[j] <= 0:
            #     continue
            # if len(words_input) + 1 == sent_index_to_id[j]:
            #     words_input.append(sent_hidden[j])
            if len(words_input) + 1 == sent_index_to_id[j + 1]:  # j + 1 is new word
                words_input.append(torch.concat((sent_hidden[j], sent_word_emb[j + 1]), dim = 0))
            
        batch_words_input.append(words_input)
    return batch_words_input

def argument_alignment(sent_hidden, sent_index_to_id):    # arguments
    words_input = []
    word_input = torch.zeros(sent_hidden[0].shape).to(device)
    temp_index_id = 1
    temp_len = 0
    for j in range(len(sent_index_to_id)):
        if sent_index_to_id[j] <= 0:
            continue
        if sent_index_to_id[j] == temp_index_id:
            word_input = word_input + sent_hidden[j]
            temp_len += 1
        else:
            words_input.append(word_input/temp_len)
            temp_len = 1
            word_input = sent_hidden[j]
        temp_index_id = sent_index_to_id[j]
    if temp_len != 0:
        words_input.append(word_input/temp_len)
    return words_input

class Graphinfo:
    def __init__(self, degree, distance, graph, father_tag):
        self.graph = np.copy(graph)
        self.distance = np.copy(distance)
        self.degree = np.copy(degree)
        self.father_tag = np.copy(father_tag)
    
    def get_info(self):
        return np.copy(self.degree), np.copy(self.distance), np.copy(self.graph), np.copy(self.father_tag)

class BEAM:
    def __init__(self, beamsize = 300):
        self.beamgraph = []
        self.beamsize = beamsize
    
    def get_batchsize(self):
        return len(self.beamgraph)
    
    def get_graphinfo(self):
        return [(item[0], item[2], item[3], item[4], item[5]) for item in self.beamgraph]
    
    def update(self, score, counts, graph, k, v, hidden):
        # for graph, score in zip(graphlist, scorelist):
        if len(self.beamgraph) < self.beamsize:
            heapq.heappush(self.beamgraph, (score, counts, graph, k, v, hidden))
        elif score > self.beamgraph[0][0]:
            heapq.heapreplace(self.beamgraph, (score, counts, graph, k, v, hidden))
    
    def abel_to_update(self, score):
        if len(self.beamgraph) < self.beamsize:
            return True
        return score > self.beamgraph[0][0]

def update_concat(tensor1, tensor2):
    if tensor1 is not None:
        return torch.cat((tensor1, tensor2), dim = 1)
    else:
        return tensor2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if __name__ == "__main__":
    # seed_everything(42)
    counter = count()
    configure_logger('logs/BLiMP_test.log')
    logger = get_logger()
    vocab_size, pad, bos, eos, startofword_id, vocab = load_vocab('../data_process/spm_parsing/BLLIP_spm.vocab')
    torch.manual_seed(123456)
    np.random.seed(123456)
    original = []
    subtoken_begins = []
    original_length = []
    original_seq_length = []

    model_path = "models/graphlayer_small_psd_4_1:0.2_mixing_ACE_predict_ahead_graph_rel_split_5_embedknet.pt"
    beamsize = 10   # 50
    scorebeamsize = 10  # 10
    logger.info("Model path: {}".format(model_path))
    logger.info("Beam size: {}".format(beamsize))
    logger.info("Score beam size: {}".format(scorebeamsize))
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model = checkpoint['model']
    biaffine_model = checkpoint['biaffine_model']
    model.eval()
    biaffine_model.eval()
    model.to(device)
    biaffine_model.to(device)

    biaffine_model.set_temperature(1.0)
    
    sp = spm.SentencePieceProcessor(model_file='../data_process/spm_parsing/BLLIP_spm.model')
    file_list = os.listdir("/home/huangty/BLiMP_data/json/.")
    final_acc = []
    for file in file_list:
        with open("/home/huangty/BLiMP_data/json/" + file, 'r') as f:
            data = f.readlines()
            sample_list = [json.loads(ds.strip()) for ds in data]
        sample_list = random.sample(sample_list, int(len(sample_list) / 10))
        logger.info(file[:-6])
        acc = 0.0
        for idx in tqdm(range(len(sample_list))):
            examples = {"sentence_good": sample_list[idx]["sentence_good"], "sentence_bad": sample_list[idx]["sentence_bad"]}
            phen2surprisals = {}
            for phen in examples:
                encoded = sp.Encode(examples[phen], out_type=int)
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
                
                id_to_index = {}
                for i in range(len(sent_index_to_id)):
                    if sent_index_to_id[i] != -1:
                        if sent_index_to_id[i] not in id_to_index:
                            id_to_index[sent_index_to_id[i]] = [i]
                        else:
                            id_to_index[sent_index_to_id[i]].append(i)

                tokens = torch.LongTensor(encoded[:-1]).to(device).reshape(1, -1)
                graph_len = max(sent_index_to_id) + 1
                graph = np.zeros((graph_len, graph_len))
                graph_distance = np.zeros((graph_len, graph_len))
                degree_list = np.zeros(graph_len)
                father_tag = np.zeros(graph_len - 1)
                arcbeam = BEAM(beamsize)
                scores = [0.0]
                step_score = 0.0
                init_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                arcbeam.update(0.0, next(counter), init_graphinfo, None, None, None)
                # beam as a batch with same token but different graph
                for i in range(tokens.shape[1]):
                    # get temp beam
                    start_time = time.time()
                    temp_beam = arcbeam.get_graphinfo()
                    temp_score = [tup[0] for tup in temp_beam]
                    batch = arcbeam.get_batchsize()
                    mask_size = i + 1
                    attn_relpos = torch.zeros(4, batch, 1, mask_size).long().to(device)
                    attn_relpos_for_pointer = torch.zeros(4, batch, max(sent_index_to_id[i], 0) + 1).long().to(device)
                    attn_relpos_for_pointer[1, :, 0] = 1 # root always depth 1
                    for step, (step_score, step_graphinfo, pre_k, pre_v, pre_hiddens) in enumerate(temp_beam):
                        degree_list, graph_distance, graph, father_tag = step_graphinfo.get_info()
                        if sent_index_to_id[i] != -1:
                            depth_list = calculate_depth(graph[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1])
                            distance_list = dijkstra(graph_distance[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1], sent_index_to_id[i])
                            for id in range(len(attn_relpos_for_pointer[0, step])):
                                attn_relpos_for_pointer[0, step, id] = degree_list[id]  # previous step graph
                            for id in range(len(attn_relpos_for_pointer[1, step])):
                                attn_relpos_for_pointer[1, step, id] = depth_list[id]
                            for id in range(len(attn_relpos_for_pointer[2, step])):
                                attn_relpos_for_pointer[2, step, id] = distance_list[id]
                            
                            for id, degree_value in enumerate(degree_list[1:]):
                                attn_relpos[0, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = degree_value
                            for id, depth_value in enumerate(depth_list[1:]):
                                attn_relpos[1, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = depth_value
                            for id, distance_value in enumerate(distance_list[1:]):
                                attn_relpos[2, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = distance_value
                            pred_depth = np.sum(father_tag[:sent_index_to_id[i]] == 0)
                            for id, father_value in enumerate(father_tag[:sent_index_to_id[i]]):
                                if father_value != 1:
                                    attn_relpos_for_pointer[3, step, id + 1] = pred_depth
                                    attn_relpos[3, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = pred_depth
                                    pred_depth -= 1
                    if i == 0:
                        cache_k = None
                        cache_v = None
                    else:
                        cache_k = torch.stack([item[2] for item in temp_beam]).view(batch, i, -1)
                        cache_v = torch.stack([item[3] for item in temp_beam]).view(batch, i, -1)
                    # rowattn_relpos =  torch.zeros(4, 1, i+1, i+1).long().to(device)
                    # batchprob = model.GraphlayerLM_inference(tokens, None, None, rowattn_relpos)
                    # [batchprob[0].view(len(encoded[1:]),-1)[idx, encoded[idx + 1]].item() for idx in range(12)]
                    prob, new_k, new_v, new_hiddens = model.GraphlayerLM_inference(tokens[:, i].repeat(batch, 1), cache_k, cache_v, attn_relpos)
                    new_hiddens = new_hiddens.transpose(0, 1)
                    transformerforward = time.time()
                    # logger.info("Transformer forward time: {}".format(transformerforward - start_time))

                    next_arcbeam = BEAM(beamsize)
                    predicate_list = []
                    arguments_list = []
                    if start_predict_new_word[i] == 1:
                        for step in range(batch):
                            step_pre_hiddens = temp_beam[step][4]
                            step_new_hiddens = new_hiddens[step:step + 1, :, :]
                            predicate = torch.concat((step_new_hiddens, model.get_emb(tokens[:, i + 1]).view(1, 1, -1)), dim=-1)
                            step_hiddens = update_concat(step_pre_hiddens, predicate)
                            predicate_list.append(step_hiddens)
                        predicates = torch.stack(predicate_list).squeeze(1)
                        with torch.no_grad():
                            graph_scores = biaffine_model.inference(predicates, attn_relpos_for_pointer)
                    pointertime = time.time()
                    # logger.info("Pointer time: {}".format(pointertime - transformerforward))
                    
                    for step in range(batch):
                        one_beam_start = time.time()
                        step_pre_hiddens = temp_beam[step][4]
                        pre_k = temp_beam[step][2]
                        pre_v = temp_beam[step][3]
                        step_new_hiddens = new_hiddens[step:step + 1, :, :]
                        predicate = torch.concat((step_new_hiddens, model.get_emb(tokens[:, i + 1]).view(1, 1, -1)), dim=-1)
                        step_k = new_k[step:step + 1, :, :]
                        step_v = new_v[step:step + 1, :, :]
                        step_hiddens = update_concat(step_pre_hiddens, predicate)

                        temp_score[step] += prob[0, step, encoded[i+1]].item()
                        stepbeam = BEAM(scorebeamsize) #(score, graphinfo)
                        cache_k = update_concat(pre_k, step_k)
                        cache_v = update_concat(pre_v, step_v)
                        stepbeam.update(temp_score[step], next(counter), temp_beam[step][1], cache_k, cache_v, step_hiddens)
                        
                        if start_predict_new_word[i] == 1:
                            # newest column
                            for j, left_score in enumerate(graph_scores[step][:-1, -1]):  # won't point itself
                                # j -> sent_index_to_id[i+1]
                                next_step_beam = BEAM(scorebeamsize)
                                previous_beam = stepbeam.get_graphinfo()  #[(score, graphinfo),()]
                                for score, graphinfo, k, v, hidden in previous_beam:
                                    # action 1 choose it
                                    new_score = score + torch.log(left_score).item()
                                    if next_step_beam.abel_to_update(new_score):
                                        degree_list, graph_distance, graph, father_tag = graphinfo.get_info()
                                        # j point to next predict word
                                        graph[j, sent_index_to_id[i+1]] = 1
                                        graph[sent_index_to_id[i+1], j] = 1
                                        graph_distance[j, sent_index_to_id[i+1]] = 10
                                        graph_distance[sent_index_to_id[i+1], j] = 1
                                        degree_list[j] += 10
                                        degree_list[sent_index_to_id[i+1]] += 1
                                        father_tag[sent_index_to_id[i+1] - 1] = 1
                                        new_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                                        next_step_beam.update(new_score, next(counter), new_graphinfo, k, v, hidden)

                                    # action 2 don't choose it
                                    new_score = score + torch.log(1 - left_score).item()
                                    if next_step_beam.abel_to_update(new_score):
                                        next_step_beam.update(new_score, next(counter), graphinfo, k, v, hidden)
                                stepbeam = next_step_beam

                            # newest row
                            for j, right_score in enumerate(graph_scores[step][-1, :-1]):
                                # sent_index_to_id[i+1] -> j
                                next_step_beam = BEAM(scorebeamsize)
                                previous_beam = stepbeam.get_graphinfo()  #[(score, graphinfo),()]
                                for score, graphinfo, k, v, hidden in previous_beam:
                                    # action 1 choose it
                                    new_score = score + torch.log(right_score).item()
                                    if next_step_beam.abel_to_update(new_score):
                                        degree_list, graph_distance, graph, father_tag = graphinfo.get_info()
                                        # next predict word point to j
                                        graph[j, sent_index_to_id[i+1]] = 1
                                        graph[sent_index_to_id[i+1], j] = 1
                                        graph_distance[j, sent_index_to_id[i+1]] = 1
                                        graph_distance[sent_index_to_id[i+1], j] = 10
                                        degree_list[j] += 1
                                        degree_list[sent_index_to_id[i+1]] += 10
                                        if j != 0:
                                            father_tag[j - 1] = 1
                                        new_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                                        next_step_beam.update(new_score, next(counter), new_graphinfo, k, v, hidden)
                                    
                                    # action 2 don't choose it
                                    new_score = score + torch.log(1 - right_score).item()
                                    if next_step_beam.abel_to_update(new_score):
                                        next_step_beam.update(new_score, next(counter), graphinfo, k, v, hidden)
                                stepbeam = next_step_beam
                        for score, graphinfo, k, v, hidden in stepbeam.get_graphinfo():
                            next_arcbeam.update(score, next(counter), graphinfo, k, v, hidden)
                        one_beam_end = time.time()
                        # logger.info("One Beam Time: {}".format(one_beam_end - one_beam_start))
                    # sum_step_scores = [scores for (scores, graphinfos) in stepbeam.get_graphinfo()]
                    step_score = -np.log(np.exp(temp_score).sum())
                    scores.append(step_score)
                    arcbeam = next_arcbeam
                    end_time = time.time()
                    # logger.info("One token Time: {}".format(end_time - start_time))
                # scores = -beam_sum_scores.cpu().numpy() # - means surprisals
                phen2surprisals[phen] = scores[-1]
            
            if phen2surprisals["sentence_good"] < phen2surprisals["sentence_bad"]:
                acc += 1
        
        logger.info(f"correct rate: {acc / len(sample_list)}")
        final_acc.append(acc / len(sample_list))
        logger.info(f"mean correct rate up to now: {np.mean(final_acc)}")
    
    logger.info(f"final correct rate: {np.mean(final_acc)}")