import numpy as np
import torch
import argparse
import os
import sys
import copy
import torch.nn as nn
import logging
import time, math
import json
from torch import cuda
from helping_utils.logger import configure_logger, get_logger
from model_bllip_dep import TransformerGrammar

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/train_LG_bllip_action.csv', type=str)
parser.add_argument('--dev_file', default='data/dev_bllip_action.csv', type=str)
parser.add_argument('--test_file', default='data/test_bllip_action.csv', type=str)
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
parser.add_argument('--rel_type', default="degree", type=str)
parser.add_argument('--eval_type', default="normal", type=str)
parser.add_argument('--sampling_num', default=0, type=int)

def log_arguments(args):

    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")

def load_data(path, batchsize=-1, shuffle=False, args=None):
    
    with open(path, 'r') as f:
        sents = [line.strip() for line in f.readlines()]
        sents = [sent.split(',') for sent in sents]
        sents = [[int(word) for word in sent] for sent in sents]
    
    if args.eval_type == "estimate":
        # repeat each sents for sampling_num times
        sents = [copy.copy(item) for item in sents for _ in range(args.sampling_num)]

    if shuffle:
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]


def get_span(hidden, index_to_id, idx):

    position = index_to_id[idx]
    span_len = 0
    if torch.cuda.is_available():
        span_hidden = torch.zeros((1024)).cuda()
    else:
        span_hidden = torch.zeros((1024))

    for word_hidden, word_index_to_id in zip(hidden, index_to_id):
        if position == word_index_to_id:
            span_hidden += word_hidden
            span_len += 1
    return span_hidden/span_len , position 


def hidden_alignment(hidden, sents_index_to_id):
    batch_words_input = []
    for sent_hidden, sent_index_to_id in zip(hidden, sents_index_to_id):
        words_input = []
        if torch.cuda.is_available():
            word_input = torch.zeros(sent_hidden[0].shape).cuda()
        else:
            word_input = torch.zeros(sent_hidden[0].shape)
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
        words_input.append(word_input/temp_len)
        batch_words_input.append(words_input)
    return batch_words_input


def load_multiarrow(path, batchsize=-1, shuffle=False, seed=1111, size="default"):
    
    with open(path, 'r') as f:
        if size == "demo":
            arrow_lists = [line.strip() for line in f.readlines()][:1000]
        elif size == "mini":
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

def eval(data, index_to_id, left_arrow, right_arrow, startofword, model, left_biaffine_model, right_biaffine_model, length, epsilon = 0, args = None):
    model.eval()
    num_sents = 0
    total_loss = 0.0
    total_loss_with_biaffine = 0.0
    num_words = 0
    uas = 0
    eval_epoch = int(args.sampling_num/args.eval_batch_size)
    count_num = 0
    setence_possible = 0.0
    sum_loss_list = [0.0]*eval_epoch
    sentence_length = 0
    loss_list = []
    bce_loss = torch.nn.BCELoss(reduction="sum")
    logger = get_logger()
    acc_num = 0
    infer_num = 0
    label_num = 0
    with torch.no_grad():
        for i in range(len(data)):
            sents = data[i]
            sents_index_to_id = index_to_id[i]
            sents_left_arrow = left_arrow[i]
            sents_right_arrow = right_arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            mems = tuple()

            # iseval for output some predictions
            # import pdb;pdb.set_trace()
            # sents[0][-30:]=[0]*30
            # sents[0][-30]=2
            # The Blair & Co. is close to an agreement to sell its TV station advertising representation operation and program production unit to an investor group led by James H. Rosenfield , a former CBS Inc. executive , industry sources said .
            ret, hidden = model(sents, startofword[i], length[i], args.attn_mask, args.document_level, True, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id, 
                        sents_arrow=[sents_left_arrow, sents_right_arrow], iseval=True, rel_type=args.rel_type)

            if args.eval_type != "estimate":               
                total_loss += ret.sum().item()
                num_words += total_length
            hidden = hidden.transpose(0, 1)
            batch_words_input = hidden_alignment(hidden, sents_index_to_id)
            biaffine_loss = 0
            batch_input1 = []
            batch_input2 = []
            batch_left_labels = []
            batch_right_labels = []
            batch_mask = []
            if [left_arrow, right_arrow]:
                for i, (sent_ids, sent_hidden, sent_index_to_id, words_input, sent_left_label, sent_right_label) in enumerate(zip(
                    sents, hidden, sents_index_to_id, batch_words_input, sents_left_arrow, sents_right_arrow)):
                    sent_biaffine_loss = 0
                    input1 = []
                    input2 = []
                    left_labels = []
                    right_labels = []
                    for j in range(len(sent_ids)):
                        if sent_index_to_id[j] != -1 and sent_index_to_id[j] != sent_index_to_id[j + 1]:    # last token of this word
                            predicate_input, words_index = get_span(sent_hidden, sent_index_to_id, j)

                            # if words_index != 1:
                            #     words_piece_input = torch.stack(words_input[:words_index - 1])
                            # elif torch.cuda.is_available():
                            #     words_piece_input = torch.Tensor([[]]).cuda()
                            # else:
                            #     words_piece_input = torch.Tensor([[]])
                            # left_logits = left_biaffine_model(predicate_input.view(1, 1, -1), words_piece_input.view(1, -1, args.w_dim)).squeeze(0,1).double()
                            # right_logits = right_biaffine_model(predicate_input.view(1, 1, -1), words_piece_input.view(1, -1, args.w_dim)).squeeze(0,1).double()
                            
                            left_gt = sent_left_label[words_index - 1]    # start from 0, 0 means root, word start from 1
                            right_gt = sent_right_label[words_index - 1]

                            max_word_length = max(sent_index_to_id)
                            padding_length = max_word_length - len(words_input[:words_index - 1])
                            words_piece_input = torch.stack(words_input[:words_index - 1] + [torch.zeros(args.w_dim).to(hidden.device)]*(padding_length))
                            
                            input1.append(predicate_input.view(1, -1))
                            input2.append(words_piece_input.view(-1, args.w_dim))

                            left_label = torch.zeros(max_word_length + 1).to(words_piece_input.device)
                            right_label = torch.zeros(max_word_length + 1).to(words_piece_input.device)
                            if left_gt:
                                left_label[left_gt] = 1
                            if right_gt:
                                right_label[right_gt] = 1
                            left_labels.append(left_label)
                            right_labels.append(right_label)

                            # # left_gt_tensor = torch.zeros(words_index).double().to(words_piece_input.device)
                            # if not left_gt:
                            #     # sent_biaffine_loss += bce_loss(left_logits, left_gt_tensor)
                            #     left_gt = list(range(words_index))
                            #     sent_biaffine_loss -= torch.sum(torch.log(1-left_logits[left_gt] + epsilon))
                            # else:
                            #     # left_gt_tensor[left_gt] = 1
                            #     # sent_biaffine_loss += bce_loss(left_logits, left_gt_tensor)
                            #     sent_biaffine_loss -= torch.sum(torch.log(left_logits[left_gt] + epsilon))
                            #     other_position = [i for i in range(words_index) if i not in left_gt]
                            #     sent_biaffine_loss -= torch.sum(torch.log(1-left_logits[other_position] + epsilon))
                            
                            # # right_gt_tensor = torch.zeros(words_index).double().to(words_piece_input.device)
                            # if not right_gt:
                            #     # sent_biaffine_loss += bce_loss(right_logits, right_gt_tensor)
                            #     right_gt = list(range(words_index))
                            #     sent_biaffine_loss -= torch.sum(torch.log(1-right_logits[right_gt] + epsilon))
                            # else:
                            #     # right_gt_tensor[right_gt] = 1
                            #     # sent_biaffine_loss += bce_loss(right_logits, right_gt_tensor)
                            #     sent_biaffine_loss -= torch.sum(torch.log(right_logits[right_gt] + epsilon))
                            #     other_position = [i for i in range(words_index) if i not in right_gt]
                            #     sent_biaffine_loss -= torch.sum(torch.log(1-right_logits[other_position] + epsilon))

                            # # if torch.isinf(biaffine_loss):
                            # #     import pdb;pdb.set_trace()
                            # #     raise ValueError("Biaffine loss is Inf.")
                            # # biaffine_loss -= torch.sum(torch.log(left_logits[left_gt])) + torch.sum(torch.log(right_logits[right_gt]))
                    if input1:
                        input1 = torch.stack(input1)
                        input2 = torch.stack(input2)
                        left_labels = torch.stack(left_labels)
                        right_labels = torch.stack(right_labels)
                        mask = torch.tril(torch.ones((max_word_length + 1, max_word_length + 1), dtype=torch.float32))[:-1,:].to(input1.device)
                        if args.eval_type != "estimate":
                            inv_left_labels = mask - left_labels
                            inv_right_labels = mask - right_labels
                            left_logits = left_biaffine_model(input1, input2).squeeze()
                            right_logits = right_biaffine_model(input1, input2).squeeze()
                            sent_biaffine_loss += - torch.sum(torch.log((1-left_logits).mul(inv_left_labels) + epsilon).mul(inv_left_labels)) - torch.sum(torch.log(left_logits.mul(left_labels) + epsilon).mul(left_labels))
                            sent_biaffine_loss += - torch.sum(torch.log((1-right_logits).mul(inv_right_labels) + epsilon).mul(inv_right_labels)) - torch.sum(torch.log(right_logits.mul(right_labels) + epsilon).mul(right_labels))
                            left_pred = torch.nonzero(left_logits*mask > 0.5, as_tuple=False)  # start from 0, 0 means root, word start from 1
                            right_pred = torch.nonzero(right_logits*mask > 0.5, as_tuple=False) 
                            left_label_nonzero = torch.nonzero(left_labels, as_tuple=False)
                            right_label_nonzero = torch.nonzero(right_labels, as_tuple=False)

                            label_num += len(left_label_nonzero)
                            label_num += len(right_label_nonzero)
                            
                            infer_num += len(left_pred) + len(right_pred)

                            for pred_row in left_pred:
                                for label_row in left_label_nonzero:
                                    if torch.equal(pred_row, label_row):  # 如果某一行完全相等
                                        acc_num += 1
                            for pred_row in right_pred:
                                for label_row in right_label_nonzero:
                                    if torch.equal(pred_row, label_row):  # 如果某一行完全相等
                                        acc_num += 1
                        else:
                            batch_input1.append(input1)
                            batch_input2.append(input2)
                            batch_left_labels.append(left_labels)
                            batch_right_labels.append(right_labels)
                            batch_mask.append(mask)
                    if args.eval_type != "estimate":
                        ret[i] += sent_biaffine_loss
            num_sents += batch_size

            if args.eval_type == "estimate":
                batch_input1 = torch.stack(batch_input1)
                batch_input2 = torch.stack(batch_input2)
                batch_left_labels = torch.stack(batch_left_labels)
                batch_right_labels = torch.stack(batch_right_labels)
                batch_mask = torch.stack(batch_mask)
                left_logits = left_biaffine_model(batch_input1, batch_input2).squeeze()
                right_logits = right_biaffine_model(batch_input1, batch_input2).squeeze()
                if left_logits.dim() == 2:
                    left_logits = left_logits.unsqueeze(1)
                    right_logits = right_logits.unsqueeze(1)
                inv_left_labels = batch_mask - batch_left_labels
                inv_right_labels = batch_mask - batch_right_labels
                biaffine_ret = -torch.sum(torch.log((1-left_logits).mul(inv_left_labels) + epsilon).mul(inv_left_labels), dim=(1,2))
                biaffine_ret += -torch.sum(torch.log(left_logits.mul(batch_left_labels) + epsilon).mul(batch_left_labels), dim=(1,2))
                biaffine_ret += -torch.sum(torch.log((1-right_logits).mul(inv_right_labels) + epsilon).mul(inv_right_labels), dim=(1,2))
                biaffine_ret += -torch.sum(torch.log(right_logits.mul(batch_right_labels) + epsilon).mul(batch_right_labels), dim=(1,2))
                ret = ret + biaffine_ret
                if count_num <= eval_epoch - 1:
                    setence_possible += np.sum(np.exp(-ret.to('cpu').numpy().astype(np.float64)))
                    loss_list.append(-math.log(setence_possible + 1e-323))
                    count_num += 1
                else:
                    total_loss_with_biaffine += -math.log(setence_possible + 1e-323)
                    assert len(loss_list) == eval_epoch
                    sum_loss_list = [x + y for x, y in zip(sum_loss_list, loss_list)]
                    count_num = 1
                    trans_ppl = np.exp(-math.log(setence_possible + 1e-323)/sentence_length)
                    logger.info(f"trans ppl {trans_ppl:.4f}, {sentence_length}")
                    setence_possible = np.sum(np.exp(-ret.to('cpu').numpy().astype(np.float64)))
                    loss_list = []
                    loss_list.append(-math.log(setence_possible + 1e-323))
                    
                if count_num == 1:
                    sentence_length = len(sents[0]) - 1
                    ori_ppl = np.exp(ret[0].item()/ sentence_length)
                    total_loss += ret[0].item()
                    num_words += sentence_length
                    logger.info(f"ori ppl {ori_ppl:.4f}")
            else:
                total_loss_with_biaffine += ret.sum().item()          
    
    ppl = np.exp(total_loss / num_words) 
    ppl_biaffine = np.exp(total_loss_with_biaffine / num_words)     
    if args.eval_type == "estimate":
        total_loss_with_biaffine += -math.log(setence_possible + 1e-323)
        ppl_biaffine = np.exp(total_loss_with_biaffine / num_words)
        trans_ppl = np.exp(-math.log(setence_possible + 1e-323)/sentence_length)
        sum_loss_list = [x + y for x, y in zip(sum_loss_list, loss_list)]
        logger.info(f"trans ppl {trans_ppl:.4f}, {sentence_length}")

        logger.info(f"sum_loss_list: {sum_loss_list}")
        logger.info(f"num of words: {num_words}")
        for i in range(len(sum_loss_list)):
            logger.info(f"Estimated sampling graph {(i+1)*args.eval_batch_size}: {np.exp(sum_loss_list[i] / num_words):.4f}")

        logger.info(f"eval ppl with biaffine loss {ppl:.4f}")
        logger.info(f"eval ppl with biaffine loss w/ sampling {ppl_biaffine:.4f}")
    else:
        logger.info(f"eval token ppl {ppl:.4f}")
        logger.info(f"eval ppl with biaffine loss w/o sampling {ppl_biaffine:.4f}")

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
        logger.info(f"pre {pre:.4f}, rec {recall:.4f}, F1 {f_1:.4f}")

    model.train()
    return ppl, uas

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev_path = args.dev_file
    test_path = args.test_file
    dev_arrow_path = args.dev_arrow_file
    test_arrow_path = args.test_arrow_file
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size

    dev_data = load_data(dev_path, batchsize=eval_batch_size, shuffle=False, args=args)
    test_data = load_data(test_path, batchsize=eval_batch_size, shuffle=False, args=args)
    left_dev_arrow, right_dev_arrow = load_multiarrow(dev_arrow_path, batchsize=eval_batch_size, shuffle=False, seed=args.seed)
    left_test_arrow, right_test_arrow = load_multiarrow(test_arrow_path, batchsize=eval_batch_size, shuffle=False, seed=args.seed)
    vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, vocab = load_vocab(args.vocab_file)

    # print(left_arc)
    # print(right_arc)
    # print(len(startofword_id))

    dev_data, startofword_dev, dev_length, dev_index_to_id = add_to_all(dev_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    test_data, startofword_test, test_length, test_index_to_id = add_to_all(test_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)

    assert len(dev_data) == len(startofword_dev)
    assert len(test_data) == len(startofword_test)
    assert len(dev_data) == len(dev_length)
    assert len(test_data) == len(test_length)
    # opening_id and closing_id are tuple-like ranges
    
    configure_logger(args.log_file)
    # log the parameters
    log_arguments(args)
    logger = get_logger()
    
    logger.info(f"dev data batches: {len(dev_data)}")
    logger.info(f"test data batches: {len(test_data)}")

    start_time = time.time()
    
    if torch.cuda.is_available():
        cuda.set_device(args.gpu)
    if args.model_file == '':
        model = TransformerGrammar(vocab_size, args.w_dim, args.n_head, args.d_head, args.d_inner, 
                                   args.num_layers, args.dropout, args.dropoutatt, pad_id, bos_id,
                                   eos_id, left_arc, right_arc, pop_root, startofword_id, args.pre_lnorm)
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
        model.apply(weights_init)
        fan_in = nn.init._calculate_correct_fan(model.emb.weight, 'fan_in')
        logger.info(f"fan in {fan_in}")
        nn.init.uniform_(model.emb.weight, -np.sqrt(3 / fan_in), np.sqrt(3 / fan_in))
    else:
        logger.info(f"loading model from {args.model_file}")
        if torch.cuda.is_available():
            checkpoint = torch.load(args.model_file)
        else:
            checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'))
        model = checkpoint['model']
        left_biaffine_model = checkpoint['left_biaffine_model']
        right_biaffine_model = checkpoint['right_biaffine_model']
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
    
    if torch.cuda.is_available():
        model.cuda()
        left_biaffine_model.cuda()
        right_biaffine_model.cuda()
    model.eval()
    left_biaffine_model.eval()
    right_biaffine_model.eval()

    if args.eval_type != "estimate":
        logger.info(f"------")
        logger.info(f"DEV SET\n")
        # logger.info(f"epsilon = 0:")       
        # val_ppl, val_uas = eval(dev_data, dev_index_to_id, left_dev_arrow, right_dev_arrow, startofword_dev, model, left_biaffine_model, right_biaffine_model, dev_length, epsilon = 0, args=args)

        logger.info(f"epsilon = 1e-323:")       
        val_ppl, val_uas = eval(dev_data, dev_index_to_id, left_dev_arrow, right_dev_arrow, startofword_dev, model, left_biaffine_model, right_biaffine_model, dev_length, epsilon = 1e-323, args=args)

    logger.info(f"------")
    logger.info(f"TEST SET\n")
    # logger.info(f"epsilon = 0:")
    # test_ppl, test_uas = eval(test_data, test_index_to_id, left_test_arrow, right_test_arrow, startofword_test, model, left_biaffine_model, right_biaffine_model, test_length, epsilon = 0, args=args)

    logger.info(f"epsilon = 1e-323:")
    test_ppl, test_uas = eval(test_data, test_index_to_id, left_test_arrow, right_test_arrow, startofword_test, model, left_biaffine_model, right_biaffine_model, test_length, epsilon = 1e-323, args=args)
 
    logger.info(f"test ppl {test_ppl:.4f}, uas {test_uas:.4f}")
    
    end_time = time.time()
    logger.info(f"total time {end_time - start_time:.2f} s")
    # logger.info(f"best val ppl {val_ppl:.4f}, uas {val_uas:.4f}")
    logger.info(f"best test ppl {test_ppl:.4f}, uas {test_uas:.4f}")
    logger.info(f"Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)