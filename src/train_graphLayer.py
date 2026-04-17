import numpy as np
import torch
import argparse
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import json
from torch import cuda
from helping_utils.logger import configure_logger, get_logger
import model_bllip_dep
from model_bllip_dep import TransformerGrammar, BiaffineAttention, find_label_idx
from scipy.stats import pearsonr, spearmanr
import csv
import math

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/train_LG_bllip_action.csv', type=str)
parser.add_argument('--dev_file', default='data/dev_bllip_action.csv', type=str)
parser.add_argument('--test_file', default='data/test_bllip_action.csv', type=str)
parser.add_argument('--train_arrow_file', default='data/train_LG_bllip_action_arrow.csv', type=str)
parser.add_argument('--dev_arrow_file', default='data/dev_bllip_action_arrow.csv', type=str)
parser.add_argument('--test_arrow_file', default='data/test_bllip_action_arrow.csv', type=str)
parser.add_argument('--log_file', default='logs/log.txt', type=str)
parser.add_argument('--model_file', default='', type=str)
parser.add_argument('--save_path', default='models/bllip.pt', type=str)
parser.add_argument('--vocab_file', default='tokenizer/spm_dp.vocab', type=str)
parser.add_argument('--sentence_level', default=False, action='store_true')
parser.add_argument('--document_level', default=False, action='store_true')
parser.add_argument('--return_h', default=False, action='store_true')
parser.add_argument('--pre_lnorm', default=False, action='store_true')
parser.add_argument('--attn_mask', default=None, type=str)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--eval_interval', default=1000, type=int)
parser.add_argument('--eval_batch_size', default=16, type=int)
parser.add_argument('--w_dim', default=384, type=int)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--d_head', default=48, type=int)
parser.add_argument('--d_inner', default=1024, type=int)
parser.add_argument('--num_layers', default=16, type=int)
parser.add_argument('--max_relative_length', default=32, type=int)
parser.add_argument('--min_relative_length', default=-32, type=int)
parser.add_argument('--seed', default=1111, type=int)
parser.add_argument('--init_std', default=0.02, type=float)
parser.add_argument('--emb_lr_multiplier', default=1.0, type=float)
parser.add_argument('--weight_decay', default=1.2e-6, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--decay_epochs', default=80, type=int)
parser.add_argument('--scheduler', default='decay', type=str, choices=['cosine', 'decay', 'const'])
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adamw'])
parser.add_argument('--lr_warm_step', default=3000, type=int)
parser.add_argument('--eta_min', default=0, type=float)
parser.add_argument('--max_lr', default=0.0003, type=float)
parser.add_argument('--start_lr', default=0.0, type=float)
parser.add_argument('--min_lr', default=0.00001, type=float)
parser.add_argument('--max_grad_norm', default=0.25, type=float)
parser.add_argument('--stable_lr', default=0.00005, type=float)
parser.add_argument('--decay_rate', default=0.5, type=float)
parser.add_argument('--decay_interval', default=2, type=int)
parser.add_argument('--log_every', default=100, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--dropoutatt', default=0.1, type=float)
parser.add_argument('--dropoute', default=0.2, type=float)
parser.add_argument('--dropouti', default=0.6, type=float)
parser.add_argument('--dropouta', default=0.2, type=float)
parser.add_argument('--dropoutf', default=0.2, type=float)
parser.add_argument('--dropouth', default=0.0, type=float)
parser.add_argument('--dropouto', default=0.5, type=float)
parser.add_argument('--alpha', default=0.2, type=float)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--transformer_lr_ratio', default=1.0, type=float)
parser.add_argument('--dataset', default="default", type=str)
parser.add_argument('--rel_type', default="degree", type=str)
parser.add_argument('--finetune', default=None, type=str)
parser.add_argument('--sts_train_path', default=None)
parser.add_argument('--sts_dev_path', default=None)
parser.add_argument('--sts_test_path', default=None)
parser.add_argument('--write_test_output', default=None)
parser.add_argument('--mixing_num', default=3, type=int)
parser.add_argument('--biaffine_head', default=1, type=int)
parser.add_argument('--biaffine_out_dim', default=1024, type=int)
parser.add_argument('--loss_alpha', default=1.0, type=float)
parser.add_argument('--loss_beta', default=1.0, type=float)
parser.add_argument('--loss_gamma', default=1.0, type=float)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def log_arguments(args):

    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")

def load_data(path, batchsize=-1, shuffle=False, seed=1111, size="default"):

    with open(path, 'r') as f:
        if size == "demo":
            sents = [line.strip() for line in f.readlines()][:1000]
        elif size == "small":
            sents = [line.strip() for line in f.readlines()]
            sents = sents[:len(sents)//4]
        else:
            sents = [line.strip() for line in f.readlines()]
        sents = [sent.split(',') for sent in sents]
        sents = [[int(word) for word in sent] for sent in sents]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]


def load_multiarrow(path, batchsize=-1, shuffle=False, seed=1111, size="default"):
    
    with open(path, 'r') as f:
        if size == "demo":
            arrow_lists = [line.strip() for line in f.readlines()][:1000]
        elif size == "small":
            arrow_lists = [line.strip() for line in f.readlines()]
            arrow_lists = arrow_lists[:len(arrow_lists)//4]
        else:
            arrow_lists = [line.strip() for line in f.readlines()]
        
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(arrow_lists)
    
    left_arc_list = [json.loads(arrow_list)["left_arc_list"] for arrow_list in arrow_lists]
    right_arc_list = [json.loads(arrow_list)["right_arc_list"] for arrow_list in arrow_lists]

    if batchsize == -1:
        return [left_arc_list], [right_arc_list]
    else:
        return [left_arc_list[i:i+batchsize] for i in range(0, len(left_arc_list), batchsize)], [right_arc_list[i:i+batchsize] for i in range(0, len(right_arc_list), batchsize)]

def load_STS_score(path, batchsize=-1, shuffle=False, seed=1111):
    with open(path, 'r') as f:
        scores = [float(line.strip()) for line in f.readlines()]
        
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(scores)
    
    if batchsize == -1:
        return [scores]
    else:
        return [scores[i:i+batchsize] for i in range(0, len(scores), batchsize)]

def add_to_all(data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id):
    
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
            arc_num = sum([1 for word in sent if word in [right_arc, left_arc]])
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
    elif finetune == "sts":
        format1 = [18737, 145, 446, 5599]
        format2 = [18737, 145, 557, 5599]
        format3 = [60, 102, 4729, 5599]
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
            

def load_vocab(path):
    vocab_file = path

    pad_id = None
    bos_id = None
    eos_id = None
    left_arc = None
    right_arc = None
    left_arc2 = None
    right_arc2 = None

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
            elif vocab[i] == '<-':
                left_arc = i
            elif vocab[i] == '->':
                right_arc = i
            elif vocab[i] == '->2':
                left_arc2 = i
            elif vocab[i] == '<-2':
                right_arc2 = i
            elif vocab[i].startswith('▁'):
                startofword_id[i] = 1

    return vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, vocab

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight'):
            scale = 2.0 / args.num_layers
            fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
            nn.init.trunc_normal_(m.weight, 0.0, np.sqrt(scale / fan_in))
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('TransformerGrammar') != -1:
        if hasattr(m, 'r_w_bias'):
            fan_in = nn.init._calculate_correct_fan(m.r_w_bias, 'fan_in')
            nn.init.trunc_normal_(m.r_w_bias, 0.0, np.sqrt(1.0 / fan_in))
        if hasattr(m, 'r_r_bias'):
            fan_in = nn.init._calculate_correct_fan(m.r_r_bias, 'fan_in')
            nn.init.trunc_normal_(m.r_r_bias, 0.0, np.sqrt(1.0 / fan_in))

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

def eval(data, index_to_id, left_arrow, right_arrow, startofword, model,
        biaffine_model, length, left_arc, right_arc, args = None, score = None,
        write_test_output = None, tape = None):
    model.eval()
    biaffine_model.eval()
    # right_biaffine_model.eval()
    num_sents = 0
    total_loss = 0.0
    biaffine_loss = 0.0
    action_loss = 0.0
    num_words = 0
    acc_num = 0
    infer_num = 0
    label_num = 0
    finetune_microf1 = 0
    finetune_pre = 0
    finetune_infer = 0
    finetune_label = 0
    crit = nn.BCEWithLogitsLoss(reduction="none")
    crit2 = nn.CrossEntropyLoss(reduction="none")
    arc_acc = 0
    action_num = 0
    topk_infer_num = 0
    topk_acc_num = 0
    prediction = []
    score_label = []
    if write_test_output is not None:
        head_data = [["index", "prediction"]]
        count = 0
    with torch.no_grad():
        for i in range(len(data)):
            sents = data[i]
            sents_index_to_id = index_to_id[i]
            sents_left_arrow = left_arrow[i]
            sents_right_arrow = right_arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            mems = tuple()
            
            ret, hidden, word_emb, attn_relpos_for_pointer, prob = model(sents, startofword[i], length[i], args.attn_mask, args.document_level, args.return_h, False, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id,
                        sents_arrow=[sents_left_arrow, sents_right_arrow], rel_type=args.rel_type, finetune=args.finetune)

            if args.finetune == "sst2":
                # 26858,5599 -> 2056: positive 2060: negative
                out = np.argmax(prob.cpu(), axis=1)
                for j in range(len(sents)):
                    idx = find_label_idx(sents[j], [26858, 5599]) + 1
                    if sents[j][idx + 1] == 2060:   # 0
                        if prob[idx][2056][j] < prob[idx][2060][j]:
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                    elif sents[j][idx + 1] == 2056:  # 1
                        if prob[idx][2056][j] > prob[idx][2060][j]:
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1

            if args.finetune == "mrpc":
                # 2745,11346,5599 -> 2064: equivalent 72 + 8864: inequivalent
                out = np.argmax(prob.cpu(), axis=1)
                for j in range(len(sents)):
                    idx = find_label_idx(sents[j], [2745, 11346, 5599]) + 2
                    if sents[j][idx + 1] == 72:   # 0
                        if prob[idx][2064][j] < prob[idx][72][j]:
                        # if sents[j][idx] == out[idx-1][j].item() and sents[j][idx - 1] == out[idx - 2][j].item():
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1
                        else:
                            finetune_infer += 1
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                    elif sents[j][idx + 1] == 2064:  # 1
                        finetune_label += 1
                        if prob[idx][2064][j] > prob[idx][72][j]:
                        # if sents[j][idx] == out[idx-1][j].item():
                            finetune_pre += 1
                            finetune_infer += 1
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1
            
            if args.finetune == "rte":
                # 2745,11346,5599 -> 221: 1 60: 0
                out = np.argmax(prob.cpu(), axis=1)
                for j in range(len(sents)):
                    idx = find_label_idx(sents[j], [2745, 11346, 5599]) + 2
                    if sents[j][idx + 1] == 60:   # 0
                        if prob[idx][221][j] < prob[idx][60][j]:
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, "not_entailment"])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, "entailment"])
                                count += 1
                    elif sents[j][idx + 1] == 221:  # 1
                        if prob[idx][221][j] > prob[idx][60][j]:
                            finetune_microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, "entailment"])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, "not_entailment"])
                                count += 1
            if args.finetune == "sts":
                out = torch.clip(ret, 0, 5).flatten().cpu().tolist()
                prediction.extend(out)
                score_label.extend(score[i])
                if write_test_output is not None:
                    for idx in range(len(sents)):
                        head_data.append([count, f"{out[idx]:.3f}"])
                        count += 1

            if args.finetune is None:
                hidden = hidden.transpose(0, 1) # non_biaffine
                word_emb = word_emb.transpose(0, 1)
                batch_predicates_input = predicate_alignment(hidden, sents_index_to_id, word_emb)
                if [left_arrow, right_arrow]:
                    for sent_ids, sent_hidden, sent_index_to_id, predicates_input, sent_left_label, sent_right_label, sent_attn_relpos in zip(
                        sents, hidden, sents_index_to_id, batch_predicates_input, sents_left_arrow, sents_right_arrow, attn_relpos_for_pointer):
                        sent_biaffine_loss = 0
                        max_word_length = max(sent_index_to_id)
                        labels = torch.zeros(max_word_length + 1, max_word_length + 1).to(device)
                        arc_num_labels = torch.zeros(max_word_length).to(device)
                        for idx, (left_label, right_label) in enumerate(zip(sent_left_label, sent_right_label)):
                            labels[idx + 1, right_label] = 1
                            labels[left_label, idx + 1] = 1
                            arc_num_labels[idx] = len(left_label) + len(right_label)
                    
                        if predicates_input:
                            scores, logits, arc_prob, arc_logits = biaffine_model.forward(torch.stack(predicates_input).unsqueeze(0), sent_attn_relpos.unsqueeze(1))
                            labels = labels[:,1:].unsqueeze(0)
                            arc_num_labels = arc_num_labels.long().unsqueeze(0)
                            sent_biaffine_loss = crit(logits, labels).sum()
                            sent_action_loss = crit2(arc_logits.permute(0,2,1), arc_num_labels).sum()
                            
                            arc_num_prediction = torch.argmax(arc_prob, dim=-1)
                            arc_scores = scores.clone()
                            for idx, (arc_pred, arc_true) in enumerate(zip(arc_num_prediction[0], arc_num_labels[0])):
                                if arc_pred == arc_true:
                                    arc_acc += 1
                                topk_infer_num += arc_pred
                                temp_scores = arc_scores[:, :(idx+2), :(idx+1)]
                                topk_values, topk_indices = torch.topk(temp_scores.flatten(), k=min(arc_pred, 2 * (idx + 1)))
                                rows = topk_indices // temp_scores.shape[2]
                                cols = topk_indices % temp_scores.shape[2]
                                arc_scores[:, :(idx+2), :(idx+1)] = 0
                                for row, col in zip(rows, cols):
                                    if labels[0, row, col] == 1:
                                        topk_acc_num += 1
                                
                            action_num += len(arc_num_labels[0])      
                            pred = torch.nonzero(scores > 0.5, as_tuple=False)
                            label_nonzero = torch.nonzero(labels, as_tuple=False)

                            label_num += len(label_nonzero)
                            
                            infer_num += len(pred)
                            for pred_row in pred:
                                for label_row in label_nonzero:
                                    if torch.equal(pred_row, label_row):
                                        acc_num += 1
         
                            biaffine_loss += sent_biaffine_loss.item()
                            action_loss += sent_action_loss.item()
            num_words += total_length
            num_sents += batch_size
            total_loss += ret.sum().item()
        
    if write_test_output is not None:
        fw = open(write_test_output, 'w')
        writer = csv.writer(fw, delimiter="\t")
        writer.writerows(head_data)
    logger = get_logger()
    if args.finetune is None:
        ppl = np.exp(total_loss / num_words) 
        biaffine_ppl = np.exp((biaffine_loss + total_loss) / num_words) # biaffineonly
        action_ppl = np.exp((action_loss + total_loss + biaffine_loss) / num_words)
        if infer_num == 0:
            f_1 = 0
            logger.info("Infer nothing...")
            pre = 0
            recall = 0
        else:
            pre = acc_num / infer_num
            recall = acc_num / label_num
            if pre == 0 or recall == 0:
                f_1 = 0
            else:
                f_1 = 2 * pre * recall / (pre + recall)
        if topk_infer_num != 0:
            topk_pre = topk_acc_num / topk_infer_num
            topk_recall = topk_acc_num / label_num
            if topk_pre == 0 or topk_recall == 0:
                topk_f1 = 0
            else:
                topk_f1 = 2 * topk_pre * topk_recall / (topk_pre + topk_recall)
        else:
            topk_f1 = 0
            topk_pre = 0
            topk_recall = 0
        logger.info(f"topk pre {topk_pre:.4f}, topk rec {topk_recall:.4f}, topk f1 {topk_f1:.4f}")
        # logger.info(f"eval token loss {total_loss / num_words:.4f}, biffine ppl {biaffine_loss.item() / num_words:.4f}")
        logger.info(f"eval token ppl {ppl:.4f}, +scoring ppl {biaffine_ppl:.4f}, +action ppl {action_ppl:.4f}")
        logger.info(f"pre {pre:.4f}, rec {recall:.4f}, F1 {f_1:.4f}")
        logger.info(f"action num acc {arc_acc / action_num:.4f}")  
    model.train()
    biaffine_model.train()
    # right_biaffine_model.train()
    if args.finetune is None:
        return action_ppl, topk_f1
    elif args.finetune == "sst2" or args.finetune == "rte":
        microf1 = finetune_microf1 / num_sents
        logger.info(f"{args.finetune} f1 {microf1:.4f}")
        return -microf1, 0
    elif args.finetune == "mrpc":
        precision = finetune_pre / finetune_infer if finetune_infer != 0 else 0
        recall = finetune_pre / finetune_label if finetune_label != 0 else 0
        logger.info(f"MRPC acc {finetune_microf1 / num_sents:.4f}")
        microf1 = 2 * (precision*recall)/(precision+recall) if precision != 0 and recall != 0 else 0
        logger.info(f"MRPC f1 {microf1:.4f}")
        return -microf1, 0
    elif args.finetune == "sts":
        prediction = np.array(prediction).reshape(-1)
        score_label = np.array(score_label).reshape(-1)
        r, _ = pearsonr(prediction, score_label)
        spr, _ = spearmanr(prediction, score_label)
        logger.info(f"---pearson correlation coefficient:{r:.4f}")
        logger.info(f"---spearman rank correlation coefficient:{spr:.4f}")
        return -(r+spr), 0


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_path = args.train_file
    dev_path = args.dev_file
    test_path = args.test_file
    train_arrow_path = args.train_arrow_file
    dev_arrow_path = args.dev_arrow_file
    test_arrow_path = args.test_arrow_file

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    model_bllip_dep.mixing_num = args.mixing_num

    train_data = load_data(train_path, batchsize=batch_size, shuffle=True, seed=args.seed, size=args.dataset)
    dev_data = load_data(dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    test_data = load_data(test_path, batchsize=eval_batch_size, shuffle=False, seed=args.seed)
    if args.finetune == "sts":
        train_sts_score = load_STS_score(args.sts_train_path, batchsize=batch_size, shuffle=True, seed=args.seed)
        dev_sts_score = load_STS_score(args.sts_dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
        test_sts_score = load_STS_score(args.sts_test_path, batchsize=eval_batch_size, shuffle=False, seed=args.seed)

    left_train_arrow, right_train_arrow = load_multiarrow(train_arrow_path, batchsize=batch_size, shuffle=True, seed=args.seed, size=args.dataset)
    left_dev_arrow, right_dev_arrow = load_multiarrow(dev_arrow_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    left_test_arrow, right_test_arrow = load_multiarrow(test_arrow_path, batchsize=eval_batch_size, shuffle=False, seed=args.seed)

    vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, vocab = load_vocab(args.vocab_file)
    train_data, startofword_train, train_length, train_index_to_id = add_to_all(train_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    dev_data, startofword_dev, dev_length, dev_index_to_id = add_to_all(dev_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    test_data, startofword_test, test_length, test_index_to_id = add_to_all(test_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)

    if args.finetune:
        train_index_to_id = add_format(train_data, train_index_to_id, args.finetune)
        dev_index_to_id = add_format(dev_data, dev_index_to_id, args.finetune)
        test_index_to_id = add_format(test_data, test_index_to_id, args.finetune)

    assert len(train_data) == len(startofword_train)
    assert len(dev_data) == len(startofword_dev)
    assert len(test_data) == len(startofword_test)
    assert len(train_data) == len(train_length)
    assert len(dev_data) == len(dev_length)
    assert len(test_data) == len(test_length)
    
    configure_logger(args.log_file)
    # log the parameters
    log_arguments(args)
    logger = get_logger()
    
    logger.info(f"train data batches: {len(train_data)}")
    logger.info(f"dev data batches: {len(dev_data)}")
    logger.info(f"test data batches: {len(test_data)}")

    start_time = time.time()
    
    if torch.cuda.device_count() == 1:
        cuda.set_device(args.gpu)
    
    if args.model_file == '':
        model = TransformerGrammar(vocab_size, args.w_dim, args.n_head, args.d_head, args.d_inner, 
                                   args.num_layers, args.dropout, args.dropoutatt, pad_id, bos_id,
                                   eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id,
                                   args.pre_lnorm, args.rel_type)
        biaffine_model = BiaffineAttention(args.biaffine_out_dim, args.w_dim, n_head=args.biaffine_head, type="Multi")
        logger.info(f"Transformer parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in biaffine_model.parameters())}")
        model.apply(weights_init)
        fan_in = nn.init._calculate_correct_fan(model.emb.weight, 'fan_in')
        logger.info(f"fan in {fan_in}")
        nn.init.uniform_(model.emb.weight, -np.sqrt(3 / fan_in), np.sqrt(3 / fan_in))
    else:
        logger.info(f"loading model from {args.model_file}")
        if torch.cuda.is_available():
            checkpoint = torch.load(args.model_file, weights_only=False)
        else:
            checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'), weights_only=False)
        if 'biaffine_model' in checkpoint:
            biaffine_model = checkpoint['biaffine_model']
            logger.info("Load exist biaffine model")
        else:
            biaffine_model = BiaffineAttention(args.biaffine_out_dim, args.w_dim, n_head=args.biaffine_head, type="Multi")
        model = checkpoint['model']
        if args.finetune is not None:
            new_model = TransformerGrammar(vocab_size, args.w_dim, args.n_head, args.d_head, args.d_inner, 
                                   args.num_layers, args.dropout, args.dropoutatt, pad_id, bos_id,
                                   eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id,
                                   args.pre_lnorm)
            new_model.load_state_dict(model.state_dict(), strict=False)
            model = new_model
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in biaffine_model.parameters())}")    
    nonemb_params = [p for p in model.parameters() if p.size() != (vocab_size, args.w_dim)]
    # nonemb_params.extend([p for p in biaffine_model.parameters()])
    emb_params = list(model.emb.parameters())
    param_list = [nonemb_params, emb_params]
    t_ratio = args.transformer_lr_ratio
    lr_list = [t_ratio, t_ratio*args.emb_lr_multiplier]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        biaffine_optimizer = torch.optim.Adam([{'params': biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        biaffine_optimizer = torch.optim.SGD([{'params': biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        biaffine_optimizer = torch.optim.AdamW([{'params': biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    crit = nn.BCEWithLogitsLoss(reduction="none")
    crit2 = nn.CrossEntropyLoss(reduction="none")
    # crit = nn.NLLLoss(reduction="mean")

    total_steps = len(train_data) * args.num_epochs
    decay_steps = len(train_data) * args.decay_epochs
    warm_up_step = args.lr_warm_step
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        biaffine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(biaffine_optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    elif args.scheduler == 'decay':
        biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.max_lr
    
    if args.model_file != '' and args.finetune is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        warm_up_scheduler.load_state_dict(checkpoint['warm_up_scheduler'])
        biaffine_optimizer.load_state_dict(checkpoint['biaffine_optimizer'])
        biaffine_scheduler.load_state_dict(checkpoint['biaffine_scheduler'])
        biaffine_warm_up_scheduler.load_state_dict(checkpoint['biaffine_warm_up_scheduler'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        for state in biaffine_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    if args.finetune == "sts":
        model.STS = nn.Linear(args.w_dim, 1)
        # nn.init.xavier_uniform_(model.STS.weight)
        # nn.init.zeros_(model.STS.bias)
        # optimizer = torch.optim.AdamW(model.STS.parameters(), lr=1e-5)
        old_optimizer_dict = checkpoint['optimizer']
        old_state = old_optimizer_dict['state']
        old_param_groups = old_optimizer_dict['param_groups']
        transformer_params = [p for p in model.parameters() if p.size() != (1, args.w_dim) and p.size() != (1,)]
        linear_params = list(model.STS.parameters())
        param_list = [transformer_params, linear_params]
        lr_list = [1, 1]
        # optimizer = torch.optim.AdamW([{'params':model.STS.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        new_param_groups = optimizer.param_groups

        for group in new_param_groups:
            for param in group['params']:
                if param in old_state:
                    optimizer.state[param] = old_state[param]
                else:
                    optimizer.state[param] = {}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()
        biaffine_model.cuda()
    model.train()
    biaffine_model.train()

    best_val_ppl = 1e5
    best_val_f1 = 0
    train_step = 0
    remaining_epoch = 0

    logger.info(f"pointer temperature: {biaffine_model.temperature}")
    checkpoint_step = 0
    mseloss = nn.MSELoss(reduction='none')
    for epoch in range(args.num_epochs):
        logger.info(f"epoch {epoch+1}")
        num_words = 0
        num_sents = 0
        train_loss = 0.0
        train_biaffine_loss = 0.0
        file_index = -1
        for i in range(len(train_data)):
            if train_step < checkpoint_step:
                # checkpoint_step -= 1
                train_step += 1

                # optimizer.zero_grad()
                # left_biaffine_optimizer.zero_grad()
                # right_biaffine_optimizer.zero_grad()
                # optimizer.step()
                # left_biaffine_optimizer.step()
                # right_biaffine_optimizer.step()
                # if args.scheduler == 'const':
                #     pass
                # elif train_step < warm_up_step:
                #     warm_up_scheduler.step()
                #     left_biaffine_warm_up_scheduler.step()
                #     right_biaffine_warm_up_scheduler.step()
                # elif args.scheduler == 'cosine':
                #     if train_step < decay_steps:
                #         scheduler.step()
                #         left_biaffine_scheduler.step()
                #         right_biaffine_scheduler.step()
                #     else:
                #         for j in range(len(optimizer.param_groups)):
                #             optimizer.param_groups[j]['lr'] = args.stable_lr
                #         for j in range(len(left_biaffine_optimizer.param_groups)):
                #             left_biaffine_optimizer.param_groups[j]['lr'] = args.stable_lr
                #         for j in range(len(right_biaffine_optimizer.param_groups)):
                #             right_biaffine_optimizer.param_groups[j]['lr'] = args.stable_lr
                continue
            tmp_time = time.time()
            sents = train_data[i]
            sents_index_to_id = train_index_to_id[i]
            sents_left_arrow = left_train_arrow[i]
            sents_right_arrow = right_train_arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            optimizer.zero_grad()
            biaffine_optimizer.zero_grad()
            mems = tuple()
            model : TransformerGrammar
            
            ret, hidden, word_emb, attn_relpos_for_pointer, _ = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id,
                        sents_arrow=[sents_left_arrow, sents_right_arrow], rel_type=args.rel_type, finetune=args.finetune)
            if args.finetune == "sts":
                sts_scores = ret
                sts_label = torch.tensor(train_sts_score[i]).unsqueeze(1).to(device)
                ret = mseloss(sts_scores, sts_label)
            biaffine_start = time.time()
            # logger.info(f"transformer foward takes {time.time()-tmp_time:.4f} seconds")
            if args.finetune is None: # non_biaffine
                hidden = hidden.transpose(0, 1)
                word_emb = word_emb.transpose(0, 1)
                batch_predicates_input = predicate_alignment(hidden, sents_index_to_id, word_emb)

                biaffine_loss = []
                if [sents_left_arrow, sents_right_arrow]:
                    for sent_ids, sent_index_to_id, predicates_input, sent_left_label, sent_right_label, attn_rel in zip(
                        sents, sents_index_to_id, batch_predicates_input, sents_left_arrow, sents_right_arrow, attn_relpos_for_pointer):
                        max_word_length = max(sent_index_to_id)
                        labels = torch.zeros(max_word_length + 1, max_word_length + 1) # max_word_length
                        arc_num_labels = torch.zeros(max_word_length)
                        for idx, (left_label, right_label) in enumerate(zip(sent_left_label, sent_right_label)):
                            labels[idx + 1, right_label] = 1
                            labels[left_label, idx + 1] = 1
                            arc_num_labels[idx] = len(left_label) + len(right_label)

                        if predicates_input:
                            labels = (labels[:,1:]).unsqueeze(0).to(device)
                            arc_num_labels = arc_num_labels.unsqueeze(0).long().to(device)
                            scores, logits, arc_prob, arc_logits = biaffine_model.forward(torch.stack(predicates_input).unsqueeze(0), attn_rel.unsqueeze(1))
                            # _, _, _ = biaffine_model.inference(torch.stack(predicates_input)[:3].unsqueeze(0), attn_rel[:, 2,:3].unsqueeze(1))
                            edge_loss = crit(logits, labels).sum()
                            action_loss = crit2(arc_logits.permute(0,2,1), arc_num_labels).sum()
                            loss = args.loss_beta * edge_loss + args.loss_gamma * action_loss
                            biaffine_loss.append(loss)
                            train_biaffine_loss += edge_loss.item() + action_loss.item()

            biaffine_end = time.time()
            # logger.info(f"Biaffine forwarding takes {biaffine_end-biaffine_start:.2f} seconds")
            raw_loss = ret
            if args.finetune is None: # non_biaffine
                raw_biaffine_loss = torch.stack(biaffine_loss)
                scale = (args.loss_beta + args.loss_gamma) / 2
                loss = 2 * (args.loss_alpha/(args.loss_alpha + scale) * raw_loss.mean() + 
                    2 / (1 + scale) * raw_biaffine_loss.mean())
            else:
                loss = raw_loss.mean()
            train_loss += raw_loss.sum().item()
            loss.backward()
            backward_time = time.time()
            # logger.info(f"Backward takes {backward_time-biaffine_end:.4f} seconds")
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(biaffine_model.parameters(), args.max_grad_norm)
            optimizer.step()
            biaffine_optimizer.step()
            train_step += 1
            if args.scheduler == 'const':
                pass
            elif train_step < warm_up_step:
                warm_up_scheduler.step()
                biaffine_warm_up_scheduler.step()
            elif args.scheduler == 'cosine':
                if train_step < decay_steps:
                    scheduler.step()
                    biaffine_scheduler.step()
                else:
                    for j in range(len(optimizer.param_groups)):
                        optimizer.param_groups[j]['lr'] = args.stable_lr
                    for j in range(len(biaffine_optimizer.param_groups)):
                        biaffine_optimizer.param_groups[j]['lr'] = args.stable_lr
            num_words += total_length
            num_sents += batch_size

            if train_step % args.log_every == 0:    #, B lr {biaffine_optimizer.param_groups[0]['lr']:.6f}
                logger.info(f"train step {train_step}, T lr {optimizer.param_groups[0]['lr']:.6f}, T loss {train_loss / num_words:.4f}, B loss {train_biaffine_loss / num_words:.4f}")
                if (train_loss + train_biaffine_loss) / num_words <= 30:
                    logger.info(f"token ppl {np.exp(train_loss / num_words):.4f}, total ppl {np.exp((train_loss + train_biaffine_loss) / num_words):.4f}")
                num_words = 0
                num_sents = 0
                train_loss = 0.0  
                train_biaffine_loss = 0.0

                # logger.info(f"dev data evaluation ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
            
            if train_step % args.eval_interval == 0 or i == len(train_data) - 1:
                if args.finetune != "sts":
                    val_ppl, val_f1 = eval(dev_data, dev_index_to_id, left_dev_arrow, right_dev_arrow, startofword_dev, model,
                    biaffine_model, dev_length, left_arc, right_arc, args=args)
                else:
                    val_ppl, val_f1 = eval(dev_data, dev_index_to_id, left_dev_arrow, right_dev_arrow, startofword_dev, model,
                    biaffine_model, dev_length, left_arc, right_arc, args=args, score=dev_sts_score)

                if val_ppl < best_val_ppl:
                    remaining_epoch = 0
                    best_val_ppl = val_ppl
                    best_val_f1 = val_f1
                    logger.info(f"new best ppl {best_val_ppl:.4f}, f1 {best_val_f1:.4f}")
                    checkpoint = {'args': args,
                                'model': model,
                                'biaffine_model': biaffine_model,
                                'vocab': vocab,
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'warm_up_scheduler': warm_up_scheduler.state_dict(),
                                'biaffine_optimizer': biaffine_optimizer.state_dict(),
                                'biaffine_warm_up_scheduler': biaffine_warm_up_scheduler.state_dict(),
                                'biaffine_scheduler': biaffine_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                }
                    try:
                        torch.save(checkpoint, args.save_path)
                    except Exception as e:
                        logger.info(f"fail to save the model")
                        if os.path.exists(args.save_path):
                            os.remove(args.save_path)
                        try:
                            torch.save(checkpoint, args.save_path)
                        except:
                            logger.info(f"still fail to save the model")
                    if args.finetune != "sts":
                        test_ppl, test_f1 = eval(test_data, test_index_to_id, left_test_arrow, right_test_arrow, startofword_test, model, 
                            biaffine_model, test_length, left_arc, right_arc, args=args, write_test_output=args.write_test_output)
                    else:
                        test_ppl, test_f1 = eval(test_data, test_index_to_id, left_test_arrow, right_test_arrow, startofword_test, model, 
                            biaffine_model, test_length, left_arc, right_arc, args=args, score=test_sts_score, write_test_output=args.write_test_output)
                    logger.info(f"test ppl {test_ppl:.4f}, f1 {test_f1:.4f}")
                elif args.scheduler == 'decay':
                    remaining_epoch += 1
                    if remaining_epoch >= args.decay_interval:
                        remaining_epoch = 0
                        for j in range(len(optimizer.param_groups)):
                            optimizer.param_groups[j]['lr'] = max(optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        for j in range(len(biaffine_optimizer.param_groups)):
                            biaffine_optimizer.param_groups[j]['lr'] = max(biaffine_optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        logger.info(f"decay lr to {optimizer.param_groups[0]['lr']:.6f}")
            # logger.info(f"other action takes {time.time()-backward_time:.2f} seconds")
    
    end_time = time.time()
    logger.info(f"total time {end_time - start_time:.2f} s")
    logger.info(f"best val ppl {best_val_ppl:.4f}, f1 {best_val_f1:.4f}")
    logger.info(f"best test ppl {test_ppl:.4f}, f1 {test_f1:.4f}")
    logger.info(f"model saved to {args.save_path}")
    logger.info(f"log saved to {args.log_file}")
    logger.info(f"Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)