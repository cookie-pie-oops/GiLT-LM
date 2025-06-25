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
from model_bllip_dep import TransformerGrammar, BiaffineAttention, calculate_depth, dijkstra, find_label_idx
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


def load_data(path, batchsize=-1, shuffle=False):
    
    with open(path, 'r') as f:
        sents = [line.strip() for line in f.readlines()]
        sents = [sent.split(',') for sent in sents]
        sents = [[int(word) for word in sent] for sent in sents]

    if shuffle:
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]

def add_to_all(data, vocab_size, pad_id, bos_id, eos_id, startofword_id):
    
    max_length = []
    startofword_copy = []
    index_to_id = []
    for batch in data:
        max_tmp = 0
        batch_startofword = []
        batch_index_to_id = []
        for sent in batch:
            sent.insert(0, bos_id)
            sent.append(eos_id)
            arc_num = 0
            sent_startofword = [vocab_size if startofword_id[word] == 1 else word for word in sent]
            count_num = 0

            sent_index_to_id = []
            for word_id in sent_startofword:
                if word_id in [bos_id, eos_id]:
                    sent_index_to_id.append(-1)
                elif word_id == vocab_size:
                    count_num += 1
                    sent_index_to_id.append(count_num)
                else:
                    sent_index_to_id.append(count_num)   #id start from 1

            batch_startofword.append(sent_startofword)
            batch_index_to_id.append(sent_index_to_id)
            length = len(sent) + arc_num
            if length > max_tmp:
                max_tmp = length
        startofword_copy.append(batch_startofword)
        index_to_id.append(batch_index_to_id)
        max_length.append(max_tmp)
    
    return data, startofword_copy, max_length, index_to_id


def add_format(data, index_to_id, finetune):
    # format1: 18737,145,446,5599, format2: 18737,145,557,5599, format3: 2745,11346,5599
    if finetune == "sst2":
        format2 = [26858, 5599]
        format1 = [18737, 145, 5599]
        format3 = []
    else:
        format1 = [18737, 145, 446, 5599]
        format2 = [18737, 145, 557, 5599]
        format3 = [2745, 11346, 5599]
    for i, batch in enumerate(data):
        for j, sent in enumerate(batch):
            start1 = find_label_idx(sent, format1)
            end1 = start1 + len(format1) - 1
            start2 = find_label_idx(sent, format2)
            end2 = start2 + len(format2) - 1
            start3 = find_label_idx(sent, format3)
            end3 = start3 + len(format3) - 1
            for k, sent_id in enumerate(sent):
                if start1 <= k <= end1:
                    index_to_id[i][j][k] = -1
                elif k > end1 and k < start2:
                    index_to_id[i][j][k] -= 1
                elif start2 <= k <= end2:
                    index_to_id[i][j][k] = -1
                elif k > end2 and k < start3:
                    index_to_id[i][j][k] -= 2
                elif start3 <= k:
                    index_to_id[i][j][k] = -1
    return index_to_id

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
        self.graph = graph
        self.distance = distance
        self.degree = degree
        self.father_tag = father_tag
    
    def get_info(self):
        return self.degree.clone(), self.distance.clone(), self.graph.clone(), self.father_tag.clone()
    
    @classmethod
    def from_existing(cls, degree_list, graph_distance, graph, father_tag):
        return cls(degree_list, graph_distance, graph, father_tag)

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
    
    def able_to_update(self, score):
        if len(self.beamgraph) < self.beamsize:
            return True
        return score > self.beamgraph[0][0]

def update_concat(tensor1, tensor2):
    if tensor1 is not None:
        return torch.cat((tensor1, tensor2), dim = 1)
    else:
        return tensor2

def update_beam(encoded, model, biaffine_model, start_predict_new_word, sent_index_to_id, beamsize, scorebeamsize, device, logger):
    with torch.no_grad():
        counter = count()
        tokens = torch.LongTensor(encoded[:-1]).to(device).reshape(1, -1)
        id_to_index = {}
        for i in range(len(sent_index_to_id)):
            if sent_index_to_id[i] != -1:
                if sent_index_to_id[i] not in id_to_index:
                    id_to_index[sent_index_to_id[i]] = [i]
                else:
                    id_to_index[sent_index_to_id[i]].append(i)

        graph_len = max(sent_index_to_id) + 1
        graph = torch.zeros((graph_len, graph_len))
        graph_distance = torch.zeros((graph_len, graph_len))
        degree_list = torch.zeros(graph_len)
        father_tag = torch.zeros(graph_len - 1)
        # arcbeam = BEAM(beamsize)
        arcbeam = []
        scores = [0.0]
        step_score = 0.0
        init_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
        # arcbeam.update(0.0, next(counter), init_graphinfo, None, None, None)    #init
        arcbeam.append((0.0, init_graphinfo, None, None, None))

        for i in range(tokens.shape[1]):
            # get temp beam
            start_time = time.time()
            # temp_beam = arcbeam.get_graphinfo()
            temp_beam = arcbeam
            temp_score = [tup[0] for tup in temp_beam]
            batch = len(arcbeam)
            mask_size = i + 1
            attn_relpos = torch.zeros(3, batch, 1, mask_size).long()  #ablation
            attn_relpos_for_pointer = torch.zeros(3, batch, max(sent_index_to_id[i], 0) + 1).long()  #ablation
            degree_index, depth_index, distance_index, pred_depth_index = 0, 1, 2, 3  #ablation
            attn_relpos_for_pointer[depth_index, :, 0] = 1 # root always depth 1
            for step, (step_score, step_graphinfo, pre_k, pre_v, pre_hiddens) in enumerate(temp_beam):
                degree_list, graph_distance, graph, father_tag = step_graphinfo.get_info()
                if sent_index_to_id[i] != -1:
                    depth_list = calculate_depth(graph[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1])
                    distance_list = dijkstra(graph_distance[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1], sent_index_to_id[i])  # graph_distance
                    for id in range(len(attn_relpos_for_pointer[degree_index, step])):
                        attn_relpos_for_pointer[degree_index, step, id] = degree_list[id]  # previous step graph
                    for id in range(len(attn_relpos_for_pointer[depth_index, step])):
                        attn_relpos_for_pointer[depth_index, step, id] = depth_list[id]
                    for id in range(len(attn_relpos_for_pointer[distance_index, step])):
                        attn_relpos_for_pointer[distance_index, step, id] = distance_list[id]
                    
                    for id, degree_value in enumerate(degree_list[1:]):
                        attn_relpos[degree_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = degree_value.item()
                    for id, depth_value in enumerate(depth_list[1:]):
                        attn_relpos[depth_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = depth_value
                    for id, distance_value in enumerate(distance_list[1:]):
                        attn_relpos[distance_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = distance_value
                    # pred_depth = torch.sum(father_tag[:sent_index_to_id[i]] == 0).item()
                    # for id, father_value in enumerate(father_tag[:sent_index_to_id[i]]):
                    #     if father_value != 1:
                    #         attn_relpos_for_pointer[pred_depth_index, step, id + 1] = pred_depth
                    #         attn_relpos[pred_depth_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = pred_depth
                    #         pred_depth -= 1
            if i == 0:
                cache_k = None
                cache_v = None
            else:
                cache_k = torch.stack([item[2] for item in temp_beam]).view(batch, i, -1)
                cache_v = torch.stack([item[3] for item in temp_beam]).view(batch, i, -1)
            # rowattn_relpos =  torch.zeros(4, 1, i+1, i+1).long().to(device)
            # batchprob = model.GraphlayerLM_inference(tokens, None, None, rowattn_relpos)
            # [batchprob[0].view(len(encoded[1:]),-1)[idx, encoded[idx + 1]].item() for idx in range(12)]
            prob, new_k, new_v, new_hiddens = model.GraphlayerLM_inference(tokens[:, i].repeat(batch, 1), cache_k, cache_v, attn_relpos.to(device))
            new_hiddens = new_hiddens.transpose(0, 1)
            transformerforward = time.time()
            # logger.info("Transformer forward time: {}".format(transformerforward - start_time))

            next_arcbeam = []
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
                    graph_scores, arc_num_prob, _ = biaffine_model.inference(predicates, attn_relpos_for_pointer.to(device))
                graph_scores, arc_num_prob = graph_scores.cpu(), arc_num_prob.cpu()
            pointertime = time.time()
            # logger.info("Pointer time: {}".format(pointertime - transformerforward))
            
            for step in range(batch):
                one_beam_start = time.time()
                step_hiddens = temp_beam[step][4]
                pre_k = temp_beam[step][2]
                pre_v = temp_beam[step][3]
                step_new_hiddens = new_hiddens[step:step + 1, :, :]
                if start_predict_new_word[i] == 1:
                    step_new_hiddens = torch.concat((step_new_hiddens, model.get_emb(tokens[:, i + 1]).view(1, 1, -1)), dim=-1)
                    step_hiddens = update_concat(step_hiddens, step_new_hiddens)
                step_k = new_k[step:step + 1, :, :]
                step_v = new_v[step:step + 1, :, :]

                temp_score[step] += prob[0, step, encoded[i+1]].item()
                # stepbeam = BEAM(scorebeamsize) #(score, graphinfo)
                cache_k = update_concat(pre_k, step_k)
                cache_v = update_concat(pre_v, step_v)
                # stepbeam.update(temp_score[step], next(counter), temp_beam[step][1], cache_k, cache_v, step_hiddens)
                # candidates = [(temp_score[step], temp_beam[step][1], cache_k, cache_v, step_hiddens)]
                # seperate1 = time.time()
                if start_predict_new_word[i] == 1:
                    temp_row = len(graph_scores[step, 0])
                    graph_scores[step, :(temp_row), :(temp_row - 1)] = -1
                    graph_scores[step, temp_row, temp_row - 1] = -1
                    temp_scores = graph_scores[step, :(temp_row+1), :(temp_row)]
                    sorted_scores, sorted_indices = torch.sort(temp_scores.flatten(), descending=True)
                    threshold = torch.topk(arc_num_prob[step], scorebeamsize)[0][-1]
                    rows = sorted_indices // temp_row
                    cols = sorted_indices % temp_row
                    # next_candidates = []
                    for arc_num, arc_score in enumerate(arc_num_prob[step]):
                        if arc_num > 2 * (temp_row) - 1:
                            break
                        if arc_score < threshold:
                            continue
                        new_score = temp_score[step] + torch.log(arc_score).item()
                        degree_list, graph_distance, graph, father_tag = temp_beam[step][1].get_info()
                        topkrows, topkcols = rows[:arc_num], cols[:arc_num]
                        graph[topkrows, topkcols + 1] = 1
                        graph[topkcols + 1, topkrows] = 1
                        cond = graph_distance[topkrows, topkcols+1] == 0
                        graph_distance[topkrows, topkcols+1] = torch.where(cond, 10.0, 1.0) #(cond, 10.0, 1.0)
                        graph_distance[topkcols+1, topkrows] = 1
                        degree_list[topkrows] += 1 #10
                        degree_list[topkcols+1] += 1
                        father_tag[topkcols] = 1
                        graphinfo = Graphinfo.from_existing(degree_list, graph_distance, graph, father_tag)
                        # next_candidates.append((new_score, graphinfo, cache_k, cache_v, step_hiddens))
                        next_arcbeam.append((new_score, graphinfo, cache_k, cache_v, step_hiddens))
                else:
                    next_arcbeam.append((temp_score[step], temp_beam[step][1], cache_k, cache_v, step_hiddens))
                # logger.info("seperate time: {}".format(time.time() - seperate1))
                    # import pdb;pdb.set_trace()
                    # candidates = next_candidates
                    # # new test setting
                    # j = 0
                    # next_step_beam = BEAM(scorebeamsize)
                    # previous_beam = stepbeam.get_graphinfo()  #[(score, graphinfo),()]
                    # for score, graphinfo, k, v, hidden in previous_beam:
                    #     # action 1 choose it
                    #     root_score = graph_scores[step][0, -1]
                    #     new_score = score + torch.log(root_score).item()
                    #     if next_step_beam.able_to_update(new_score):
                    #         degree_list, graph_distance, graph, father_tag = graphinfo.get_info()
                    #         # j point to next predict word
                    #         graph[j, sent_index_to_id[i+1]] = 1
                    #         graph[sent_index_to_id[i+1], j] = 1
                    #         graph_distance[j, sent_index_to_id[i+1]] = 10 if graph_distance[j, sent_index_to_id[i+1]] == 0 else 1
                    #         # graph_distance[j, sent_index_to_id[i+1]] = 10
                    #         graph_distance[sent_index_to_id[i+1], j] = 1
                    #         degree_list[j] += 10
                    #         degree_list[sent_index_to_id[i+1]] += 1
                    #         father_tag[sent_index_to_id[i+1] - 1] = 1
                    #         new_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                    #         next_step_beam.update(new_score, next(counter), new_graphinfo, k, v, hidden)

                    #     # action 2 don't choose it
                    #     new_score = score + torch.log(1 - root_score).item()
                    #     if next_step_beam.able_to_update(new_score):
                    #         next_step_beam.update(new_score, next(counter), graphinfo, k, v, hidden)
                    # stepbeam = next_step_beam

                    # for j, (left_score, right_score) in enumerate(zip(graph_scores[step][1:-1, -1], graph_scores[step][-1, :-1])):
                    #     next_step_beam = BEAM(scorebeamsize)
                    #     previous_beam = stepbeam.get_graphinfo()  #[(score, graphinfo),()]
                    #     for score, graphinfo, k, v, hidden in previous_beam:
                    #         # action 1 left
                    #         new_score = score + torch.log(left_score).item()
                    #         if next_step_beam.able_to_update(new_score):
                    #             degree_list, graph_distance, graph, father_tag = graphinfo.get_info()
                    #             # j point to next predict word
                    #             graph[j + 1, sent_index_to_id[i+1]] = 1
                    #             graph[sent_index_to_id[i+1], j + 1] = 1
                    #             # graph_distance[j + 1, sent_index_to_id[i+1]] = 10 if graph_distance[j + 1, sent_index_to_id[i+1]] == 0 else 1
                    #             graph_distance[j + 1, sent_index_to_id[i+1]] = 10
                    #             graph_distance[sent_index_to_id[i+1], j + 1] = 1
                    #             degree_list[j + 1] += 10
                    #             degree_list[sent_index_to_id[i+1]] += 1
                    #             father_tag[sent_index_to_id[i+1] - 1] = 1
                    #             new_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                    #             next_step_beam.update(new_score, next(counter), new_graphinfo, k, v, hidden)
                            
                    #         # action 2 right
                    #         new_score = score + torch.log(right_score).item()
                    #         if next_step_beam.able_to_update(new_score):
                    #             degree_list, graph_distance, graph, father_tag = graphinfo.get_info()
                    #             # next predict word point to j
                    #             graph[j + 1, sent_index_to_id[i+1]] = 1
                    #             graph[sent_index_to_id[i+1], j + 1] = 1
                    #             graph_distance[j + 1, sent_index_to_id[i+1]] = 1
                    #             # graph_distance[sent_index_to_id[i+1], j + 1] = 10 if graph_distance[sent_index_to_id[i+1], j + 1] == 0 else 1
                    #             graph_distance[sent_index_to_id[i+1], j + 1] = 10
                    #             degree_list[j + 1] += 1
                    #             degree_list[sent_index_to_id[i+1]] += 10
                    #             father_tag[j] = 1
                    #             new_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
                    #             next_step_beam.update(new_score, next(counter), new_graphinfo, k, v, hidden)

                    #         # action 3 don't choose any
                    #         new_score = score + torch.log(1 - left_score).item() + torch.log(1 - right_score).item()
                    #         if next_step_beam.able_to_update(new_score):
                    #             next_step_beam.update(new_score, next(counter), graphinfo, k, v, hidden)
                    #     stepbeam = next_step_beam
                # for score, graphinfo, k, v, hidden in stepbeam.get_graphinfo():
                # for score, graphinfo, k, v, hidden in candidates:
                #     # next_arcbeam.update(score, next(counter), graphinfo, k, v, hidden)
                #     next_arcbeam.append((score, graphinfo, k, v, hidden))
                one_beam_end = time.time()
            # logger.info("One beam time: {}".format(one_beam_end - one_beam_start))
            next_arcbeam.sort(key=lambda x: -x[0])
            next_arcbeam = next_arcbeam[:beamsize]
            step_score = -(np.max(temp_score) + np.log(np.sum(np.exp(temp_score - np.max(temp_score)))))
            scores.append(step_score)
            arcbeam = next_arcbeam
            # logger.info("one step time: {}".format(time.time() - start_time))
    
    return scores, arcbeam


def GiLT_GPT2_update_beam(encoded, model, biaffine_model, start_predict_new_word, sent_index_to_id, beamsize, scorebeamsize, device, logger):
    with torch.no_grad():
        counter = count()
        tokens = torch.LongTensor(encoded[:-1]).to(device).reshape(1, -1)
        id_to_index = {}
        for i in range(len(sent_index_to_id)):
            if sent_index_to_id[i] != -1:
                if sent_index_to_id[i] not in id_to_index:
                    id_to_index[sent_index_to_id[i]] = [i]
                else:
                    id_to_index[sent_index_to_id[i]].append(i)

        graph_len = max(sent_index_to_id) + 1
        graph = torch.zeros((graph_len, graph_len))
        graph_distance = torch.zeros((graph_len, graph_len))
        degree_list = torch.zeros(graph_len)
        father_tag = torch.zeros(graph_len - 1)
        # arcbeam = BEAM(beamsize)
        arcbeam = []
        scores = [0.0]
        step_score = 0.0
        init_graphinfo = Graphinfo(degree_list, graph_distance, graph, father_tag)
        # arcbeam.update(0.0, next(counter), init_graphinfo, None, None, None)    #init
        arcbeam.append((0.0, init_graphinfo, None, None, None))
        for i in range(tokens.shape[1]):
            # get temp beam
            start_time = time.time()
            # temp_beam = arcbeam.get_graphinfo()
            temp_beam = arcbeam
            temp_score = [tup[0] for tup in temp_beam]
            batch = len(arcbeam)
            mask_size = i + 1
            attn_relpos = torch.zeros(3, batch, 1, mask_size).long()
            attn_relpos_for_pointer = torch.zeros(3, batch, max(sent_index_to_id[i], 0) + 1).long()
            degree_index, depth_index, distance_index = 0, 1, 2
            attn_relpos_for_pointer[depth_index, :, 0] = 1 # root always depth 1
            for step, (step_score, step_graphinfo, pre_k, pre_v, pre_hiddens) in enumerate(temp_beam):
                degree_list, graph_distance, graph, father_tag = step_graphinfo.get_info()
                if sent_index_to_id[i] != -1:
                    depth_list = calculate_depth(graph[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1])
                    distance_list = dijkstra(graph_distance[:sent_index_to_id[i] + 1, :sent_index_to_id[i] + 1], sent_index_to_id[i])  # graph_distance
                    for id in range(len(attn_relpos_for_pointer[degree_index, step])):
                        attn_relpos_for_pointer[degree_index, step, id] = degree_list[id]  # previous step graph
                    for id in range(len(attn_relpos_for_pointer[depth_index, step])):
                        attn_relpos_for_pointer[depth_index, step, id] = depth_list[id]
                    for id in range(len(attn_relpos_for_pointer[distance_index, step])):
                        attn_relpos_for_pointer[distance_index, step, id] = distance_list[id]
                    
                    for id, degree_value in enumerate(degree_list[1:]):
                        attn_relpos[degree_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = degree_value.item()
                    for id, depth_value in enumerate(depth_list[1:]):
                        attn_relpos[depth_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = depth_value
                    for id, distance_value in enumerate(distance_list[1:]):
                        attn_relpos[distance_index, step, 0, [idx for idx in id_to_index[id + 1] if idx < mask_size]] = distance_value
            if i == 0:
                kv_cache = None
            else:
                kv_cache = []
                beams_kv = [item[2] for item in temp_beam]
                for layer_idx in range(len(beams_kv[0])):
                    all_k = [beam_kv[layer_idx][0] for beam_kv in beams_kv]
                    all_v = [beam_kv[layer_idx][1] for beam_kv in beams_kv]
                    kv_cache.append((torch.cat(all_k, dim=0), torch.cat(all_v, dim=0)))
                kv_cache = tuple(kv_cache)
            outputs = model.generate(tokens[:, i:(i+1)].repeat(batch, 1), attn_relpos=attn_relpos, use_cache=True, return_dict=True, output_hidden_states=True, past_key_values=kv_cache)
            new_hiddens = torch.cat([outputs.hidden_states[-2], outputs.hidden_states[13]], dim=-1)
            kv_cache = outputs.past_key_values
            prob = outputs.logits.log_softmax(-1)
            # new_hiddens = new_hiddens.transpose(0, 1)
            transformerforward = time.time()
            # logger.info("Transformer forward time: {}".format(transformerforward - start_time))

            next_arcbeam = []
            predicate_list = []
            arguments_list = []
            if start_predict_new_word[i] == 1:
                for step in range(batch):
                    step_pre_hiddens = temp_beam[step][4]
                    step_new_hiddens = new_hiddens[step:step + 1, :, :]
                    predicate = torch.concat((step_new_hiddens, model.transformer.wte(tokens[:, i + 1]).view(1, 1, -1)), dim=-1)
                    step_hiddens = update_concat(step_pre_hiddens, predicate)
                    predicate_list.append(step_hiddens)
                predicates = torch.stack(predicate_list).squeeze(1)
                with torch.no_grad():
                    graph_scores, arc_num_prob, _ = biaffine_model.inference(predicates, attn_relpos_for_pointer.to(device))
                graph_scores, arc_num_prob = graph_scores.cpu(), arc_num_prob.cpu()
            pointertime = time.time()
            # logger.info("Pointer time: {}".format(pointertime - transformerforward))
            for step in range(batch):
                one_beam_start = time.time()
                step_hiddens = temp_beam[step][4]
                step_new_hiddens = new_hiddens[step:step + 1, :, :]
                if start_predict_new_word[i] == 1:
                    step_new_hiddens = torch.concat((step_new_hiddens, model.transformer.wte(tokens[:, i + 1]).view(1, 1, -1)), dim=-1)
                    step_hiddens = update_concat(step_hiddens, step_new_hiddens)
                past_kv = tuple(tuple(mat[step:step + 1, :, :, :] for mat in layer_kv) for layer_kv in kv_cache)

                temp_score[step] += prob[step, 0, encoded[i+1]].item()
                # stepbeam = BEAM(scorebeamsize) #(score, graphinfo)

                # stepbeam.update(temp_score[step], next(counter), temp_beam[step][1], cache_k, cache_v, step_hiddens)
                # candidates = [(temp_score[step], temp_beam[step][1], cache_k, cache_v, step_hiddens)]
                # seperate1 = time.time()
                if start_predict_new_word[i] == 1:
                    temp_row = len(graph_scores[step, 0])
                    graph_scores[step, :(temp_row), :(temp_row - 1)] = -1
                    graph_scores[step, temp_row, temp_row - 1] = -1
                    temp_scores = graph_scores[step, :(temp_row+1), :(temp_row)]
                    sorted_scores, sorted_indices = torch.sort(temp_scores.flatten(), descending=True)
                    threshold = torch.topk(arc_num_prob[step], scorebeamsize)[0][-1]
                    rows = sorted_indices // temp_row
                    cols = sorted_indices % temp_row
                    # next_candidates = []
                    for arc_num, arc_score in enumerate(arc_num_prob[step]):
                        if arc_num > 2 * (temp_row) - 1:
                            break
                        if arc_score < threshold:
                            continue
                        new_score = temp_score[step] + torch.log(arc_score).item()
                        degree_list, graph_distance, graph, father_tag = temp_beam[step][1].get_info()
                        topkrows, topkcols = rows[:arc_num], cols[:arc_num]
                        graph[topkrows, topkcols + 1] = 1
                        graph[topkcols + 1, topkrows] = 1
                        cond = graph_distance[topkrows, topkcols+1] == 0
                        graph_distance[topkrows, topkcols+1] = torch.where(cond, 10.0, 1.0) #(cond, 10.0, 1.0)
                        graph_distance[topkcols+1, topkrows] = 1
                        degree_list[topkrows] += 10 #10
                        degree_list[topkcols+1] += 1
                        father_tag[topkcols] = 1
                        graphinfo = Graphinfo.from_existing(degree_list, graph_distance, graph, father_tag)
                        # next_candidates.append((new_score, graphinfo, cache_k, cache_v, step_hiddens))
                        next_arcbeam.append((new_score, graphinfo, past_kv, None, step_hiddens))
                else:
                    next_arcbeam.append((temp_score[step], temp_beam[step][1], past_kv, None, step_hiddens))
                # logger.info("seperate time: {}".format(time.time() - seperate1))
                one_beam_end = time.time()
            # logger.info("One beam time: {}".format(one_beam_end - one_beam_start))
            next_arcbeam.sort(key=lambda x: -x[0])
            next_arcbeam = next_arcbeam[:beamsize]
            step_score = -(np.max(temp_score) + np.log(np.sum(np.exp(temp_score - np.max(temp_score)))))
            scores.append(step_score)
            arcbeam = next_arcbeam
            # logger.info("one step time: {}".format(time.time() - start_time))
    
    return scores, arcbeam

def get_best_graph(beam):
    score_list = [item[0] for item in beam]
    return beam[np.argmax(score_list)][1].get_info()[1]

def get_action_from_graph(graph_matrix):
    left_arc_list = []
    right_arc_list = []
    for i in range(len(graph_matrix) - 1):
        row = graph_matrix[i+1, 0:i+2]
        col = graph_matrix[0:i+2, i+1]
        temp_token_left_list = []
        temp_token_right_list = []
        for index, arc_exist in enumerate(row): # i -> j
            if arc_exist == 10:
                temp_token_right_list.append(int(index)) #index start from 1
                
        for index, arc_exist in enumerate(col): # j -> i
            if arc_exist == 10:
                temp_token_left_list.append(int(index)) #index start from 1

        left_arc_list.append(temp_token_left_list)
        right_arc_list.append(temp_token_right_list)
    arrow_dict = {"left_arc_list":left_arc_list,"right_arc_list":right_arc_list}
    return arrow_dict