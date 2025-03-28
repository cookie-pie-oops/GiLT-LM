import numpy as np
import torch
import argparse
import os
import sys
import torch.nn as nn
import logging
import time
import json
from torch import cuda
from helping_utils.logger import configure_logger, get_logger
from model_bllip_dep import TransformerGrammar, BiaffineAttention

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
parser.add_argument('--proj_dim', default=128, type=int)
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
parser.add_argument('--BTloss_ratio', default=1.0, type=float)
parser.add_argument('--dataset', default="default", type=str)
parser.add_argument('--rel_type', default="degree", type=str)
parser.add_argument('--stage_two', default=False, action='store_true')
parser.add_argument('--smoothlabel', default=False, action='store_true')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

biaffine_hidden = 2 * 1024

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


def align_data_arrow(arrows, data, left_arc, right_arc):
    align_arrows = []
    arrow_id = [left_arc, right_arc]
    for i in range(len(data)):
        batch_arrows = []
        for j in range(len(data[i])):
            new_arrows = [-1]*len(data[i][j])
            count = 0
            for k in range(len(data[i][j])):
                if data[i][j][k] in arrow_id:
                    new_arrows[k] = arrows[i][j][count]
                    count += 1
            assert count == len(arrows[i][j])
            batch_arrows.append(new_arrows)
        align_arrows.append(batch_arrows)
    return align_arrows
            

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

def get_span(hidden, index_to_id, idx):

    position = index_to_id[idx]
    span_len = 0
    span_hidden = torch.zeros((biaffine_hidden)).to(device)

    for word_hidden, word_index_to_id in zip(hidden, index_to_id):
        if position == word_index_to_id:
            span_hidden += word_hidden
            span_len += 1
    return span_hidden/span_len , position 
    #span_hidden/span_len/2 + hidden[arc_pos]/2  (span_hidden + hidden[arc_pos])/(span_len + 1)

def hidden_alignment(hidden, sents_index_to_id):
    batch_words_input = []
    for sent_hidden, sent_index_to_id in zip(hidden, sents_index_to_id):
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
        words_input.append(word_input/temp_len)
        batch_words_input.append(words_input)
    return batch_words_input

def eval(data, index_to_id, left_arrow, right_arrow, startofword, model,
        left_biaffine_model, right_biaffine_model, length, left_arc, right_arc, args = None):
    model.eval()
    left_biaffine_model.eval()
    right_biaffine_model.eval()
    num_sents = 0
    total_loss = 0.0
    biaffine_loss = 0.0
    num_words = 0
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
            
            ret, hidden = model(sents, startofword[i], length[i], args.attn_mask, args.document_level, args.return_h, False, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id,
                        sents_arrow=[sents_left_arrow, sents_right_arrow], rel_type=args.rel_type)
            hidden = hidden.transpose(0, 1) # change
            batch_words_input = hidden_alignment(hidden, sents_index_to_id)
            if [left_arrow, right_arrow]:
                for sent_ids, sent_hidden, sent_index_to_id, words_input, sent_left_label, sent_right_label in zip(
                    sents, hidden, sents_index_to_id, batch_words_input, sents_left_arrow, sents_right_arrow):
                    max_word_length = max(sent_index_to_id)
                    words_piece_input = torch.stack(words_input)
                    for j in range(len(sent_ids)):
                        if sent_index_to_id[j] != -1 and sent_index_to_id[j] != sent_index_to_id[j + 1]:    # last token of this word
                            # predicate_input, words_index = get_span(sent_hidden, sent_index_to_id, j)
                            words_index = sent_index_to_id[j]
                            predicate_input = words_input[words_index - 1]
                            # if torch.cuda.is_available():
                            #     words_piece_input = torch.stack([torch.zeros(args.w_dim).cuda()] + words_input[:words_index])
                            # else:
                            #     words_piece_input = torch.stack([torch.zeros(args.w_dim)] + words_input[:words_index])
                            # if words_index != 1:
                            #     words_piece_input = torch.stack(words_input[:words_index - 1])
                            # else:
                            #     words_piece_input = torch.Tensor([[]]).to(device)
                            left_logits = left_biaffine_model(predicate_input.view(1, 1, -1), words_piece_input[:words_index - 1].view(1, -1, biaffine_hidden)).squeeze()
                            right_logits = right_biaffine_model(predicate_input.view(1, 1, -1), words_piece_input[:words_index - 1].view(1, -1, biaffine_hidden)).squeeze()
                            if left_logits.dim() == 0:
                                left_logits = left_logits.unsqueeze(0)
                                right_logits = right_logits.unsqueeze(0)
                            left_pred = torch.where(left_logits > 0.5)  # start from 0, 0 means root, word start from 1
                            right_pred = torch.where(right_logits > 0.5)
                            
                            left_gt = sent_left_label[words_index - 1]    # start from 0, 0 means root, word start from 1
                            right_gt = sent_right_label[words_index - 1]

                            if left_gt:
                                label_num += len(left_gt)
                            if right_gt:
                                label_num += len(right_gt)
                            
                            infer_num += len(left_pred[0]) + len(right_pred[0])

                            if left_pred:
                                for l in left_pred[0]:
                                    if l.item() in left_gt:
                                        acc_num += 1
                            if right_pred:
                                for r in right_pred[0]:
                                    if r.item() in right_gt:
                                        acc_num += 1
                            epsilon = 1e-323
                            if not left_gt:
                                left_gt = list(range(words_index))
                                biaffine_loss -= torch.sum(torch.log(1-left_logits[left_gt] + epsilon)).item()
                            else:
                                biaffine_loss -= torch.sum(torch.log(left_logits[left_gt] + epsilon)).item()
                                other_position = [i for i in range(words_index) if i not in left_gt]
                                biaffine_loss -= torch.sum(torch.log(1-left_logits[other_position] + epsilon)).item()
                            
                            if not right_gt:
                                right_gt = list(range(words_index))
                                biaffine_loss -= torch.sum(torch.log(1-right_logits[right_gt] + epsilon)).item()
                            else:
                                biaffine_loss -= torch.sum(torch.log(right_logits[right_gt] + epsilon)).item()
                                other_position = [i for i in range(words_index) if i not in right_gt]
                                biaffine_loss -= torch.sum(torch.log(1-right_logits[other_position] + epsilon)).item()

            num_words += total_length
            num_sents += batch_size
            total_loss += ret.sum().item()

    ppl = np.exp(total_loss / num_words) 
    biaffine_ppl = np.exp((biaffine_loss + total_loss) / num_words)
    logger = get_logger()
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
    # logger.info(f"eval token loss {total_loss / num_words:.4f}, biffine ppl {biaffine_loss.item() / num_words:.4f}")
    logger.info(f"eval token ppl {ppl:.4f}, total ppl {biaffine_ppl:.4f}")
    logger.info(f"pre {pre:.4f}, rec {recall:.4f}, F1 {f_1:.4f}")
    model.train()
    left_biaffine_model.train()
    right_biaffine_model.train()
    return biaffine_ppl, f_1


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

    train_data = load_data(train_path, batchsize=batch_size, shuffle=True, seed=args.seed, size=args.dataset)
    dev_data = load_data(dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    test_data = load_data(test_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)

    left_train_arrow, right_train_arrow = load_multiarrow(train_arrow_path, batchsize=batch_size, shuffle=True, seed=args.seed, size=args.dataset)
    left_dev_arrow, right_dev_arrow = load_multiarrow(dev_arrow_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    left_test_arrow, right_test_arrow = load_multiarrow(test_arrow_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)

    vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, vocab = load_vocab(args.vocab_file)
    # print(left_arc)
    # print(right_arc)
    # print(len(startofword_id))
    train_data, startofword_train, train_length, train_index_to_id = add_to_all(train_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    dev_data, startofword_dev, dev_length, dev_index_to_id = add_to_all(dev_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    test_data, startofword_test, test_length, test_index_to_id = add_to_all(test_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)

    # train_arrow = align_data_arrow(train_arrow, train_data, left_arc, right_arc)
    # dev_arrow = align_data_arrow(dev_arrow, dev_data, left_arc, right_arc)
    # test_arrow = align_data_arrow(test_arrow, test_data, left_arc, right_arc)

    assert len(train_data) == len(startofword_train)
    assert len(dev_data) == len(startofword_dev)
    assert len(test_data) == len(startofword_test)
    assert len(train_data) == len(train_length)
    assert len(dev_data) == len(dev_length)
    assert len(test_data) == len(test_length)
    # opening_id and closing_id are tuple-like ranges
    
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
                                   eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, args.pre_lnorm)
        left_biaffine_model = BiaffineAttention(args.d_inner, args.proj_dim, type="Multi")
        right_biaffine_model = BiaffineAttention(args.d_inner, args.proj_dim, type="Multi")
        logger.info(f"Transformer parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in left_biaffine_model.parameters()) * 2}")
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
        if 'left_biaffine_model' in checkpoint:
            left_biaffine_model = checkpoint['left_biaffine_model']
            right_biaffine_model = checkpoint['right_biaffine_model']
            logger.info("Load exist biaffine model")
        else:
            left_biaffine_model = BiaffineAttention(args.w_dim, args.proj_dim, type="Multi")
            right_biaffine_model = BiaffineAttention(args.w_dim, args.proj_dim, type="Multi")
        model = checkpoint['model']
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in left_biaffine_model.parameters()) * 2}")
    
    nonemb_params = [p for p in model.parameters() if p.size() != (vocab_size, args.w_dim)]
    emb_params = list(model.emb.parameters())
    # param_list = [biaffine_model.parameters(), nonemb_params, emb_params]
    # lr_list = [1, 1, args.emb_lr_multiplier]
    param_list = [nonemb_params, emb_params]
    t_ratio = args.transformer_lr_ratio
    lr_list = [t_ratio, t_ratio*args.emb_lr_multiplier]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        left_biaffine_optimizer = torch.optim.Adam([{'params': left_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
        right_biaffine_optimizer = torch.optim.Adam([{'params': right_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        left_biaffine_optimizer = torch.optim.SGD([{'params': left_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
        right_biaffine_optimizer = torch.optim.SGD([{'params': right_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
        left_biaffine_optimizer = torch.optim.AdamW([{'params': left_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
        right_biaffine_optimizer = torch.optim.AdamW([{'params': right_biaffine_model.parameters(), 'lr': 1}], weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    crit = nn.BCELoss(reduction="none")
    # crit = nn.NLLLoss(reduction="mean")

    total_steps = len(train_data) * args.num_epochs
    decay_steps = len(train_data) * args.decay_epochs
    warm_up_step = args.lr_warm_step
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        left_biaffine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(left_biaffine_optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        right_biaffine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(right_biaffine_optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        left_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(left_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        right_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(right_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    elif args.scheduler == 'decay':
        left_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(left_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        right_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(right_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.max_lr
    
    if args.model_file != '':
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        warm_up_scheduler.load_state_dict(checkpoint['warm_up_scheduler'])
        left_biaffine_optimizer.load_state_dict(checkpoint['left_biaffine_optimizer'])
        right_biaffine_optimizer.load_state_dict(checkpoint['right_biaffine_optimizer'])
        left_biaffine_scheduler.load_state_dict(checkpoint['left_biaffine_scheduler'])
        right_biaffine_scheduler.load_state_dict(checkpoint['right_biaffine_scheduler'])
        left_biaffine_warm_up_scheduler.load_state_dict(checkpoint['left_biaffine_warm_up_scheduler'])
        right_biaffine_warm_up_scheduler.load_state_dict(checkpoint['right_biaffine_warm_up_scheduler'])
        if torch.cuda.is_available():
            for state in left_biaffine_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            for state in right_biaffine_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if args.stage_two:
            decay_steps = len(train_data)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
            left_biaffine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(left_biaffine_optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
            right_biaffine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(right_biaffine_optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
            warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
            left_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(left_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
            right_biaffine_warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(right_biaffine_optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    
    if args.smoothlabel:
        positivelabel = 0.9
        negativelabel = 0.1
    else:
        positivelabel = 1.0
        negativelabel = 0.0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()
        left_biaffine_model.cuda()
        right_biaffine_model.cuda()
    model.train()
    left_biaffine_model.train()
    right_biaffine_model.train()

    best_val_ppl = 1e5
    best_val_f1 = 0
    train_step = 0
    remaining_epoch = 0

    # dynamic learning rate
    # log_start = np.log(0.01)
    # log_end = np.log(100)
    # slope = (log_end - log_start) / total_steps
    # BTloss_ratio_list = [np.exp(log_start + slope * step) for step in range(total_steps)]

    log_start = np.log(0.01) # change
    log_end = np.log(5)
    slope = (log_end - log_start) / total_steps * 2
    BTloss_ratio_list = [np.exp(log_start + slope * step) for step in range(total_steps // 2)] + [np.exp(log_end - slope * step) for step in range(total_steps // 2 + 1)]
    
    if args.stage_two:
        checkpoint_step = (args.num_epochs - 1) * len(train_data)
    else:
        checkpoint_step = 0
    for epoch in range(args.num_epochs):
        logger.info(f"epoch {epoch+1}")
        num_words = 0
        num_sents = 0
        train_loss = 0.0
        train_biaffine_loss = 0.0
        for i in range(len(train_data)):
            if train_step < checkpoint_step:
                checkpoint_step -= 1
                # train_step += 1

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
            # torch.cuda.empty_cache()
            tmp_time = time.time()
            sents = train_data[i]
            sents_index_to_id = train_index_to_id[i]
            sents_left_arrow = left_train_arrow[i]
            sents_right_arrow = right_train_arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            optimizer.zero_grad()
            left_biaffine_optimizer.zero_grad()
            right_biaffine_optimizer.zero_grad()
            mems = tuple()
            model : TransformerGrammar

            # model.eval()
            # with torch.no_grad():
            #     ret, hidden = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
            #             args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id, sents_arrow=sents_arrow)
            ret, hidden = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id,
                        sents_arrow=[sents_left_arrow, sents_right_arrow], rel_type=args.rel_type)

            if not args.stage_two or (args.stage_two and epoch == args.num_epochs - 1): # change
                hidden = hidden.transpose(0, 1)
                batch_words_input = hidden_alignment(hidden, sents_index_to_id)

                biaffine_loss = []
                # max_length = max([len(sent) for sent in sents])
                # max_word_length = max(max(item) for item in sents_index_to_id) + 1
                if [sents_left_arrow, sents_right_arrow]:
                    for sent_ids, sent_hidden, sent_index_to_id, words_input, sent_left_label, sent_right_label in zip(
                        sents, hidden, sents_index_to_id, batch_words_input, sents_left_arrow, sents_right_arrow):
                        input1 = []
                        max_word_length = max(sent_index_to_id)
                        words_piece_input = torch.stack(words_input)
                        input2 = torch.unsqueeze(words_piece_input, 0).repeat(max_word_length, 1, 1)
                        left_labels = torch.zeros(max_word_length, max_word_length + 1).to(device)
                        right_labels = torch.zeros(max_word_length, max_word_length + 1).to(device)
                        for j in range(len(sent_ids)):
                            if sent_index_to_id[j] != -1 and sent_index_to_id[j] != sent_index_to_id[j + 1]:    # last token of this word
                                # predicate_input, words_index = get_span(sent_hidden, sent_index_to_id, j)
                                words_index = sent_index_to_id[j]
                                words_index -= 1
                                predicate_input = words_input[words_index]
                                input1.append(predicate_input.view(1, -1))

                                if sent_left_label[words_index]:
                                    left_labels[words_index][sent_left_label[words_index]] = positivelabel 
                                if sent_right_label[words_index]:
                                    right_labels[words_index][sent_right_label[words_index]] = positivelabel

                        if input1:
                            input1 = torch.stack(input1)
                            mask = torch.tril(torch.ones((max_word_length + 1, max_word_length + 1), dtype=torch.float32))[:-1,:]
                            mask = mask.to(device)
                            left_logits = left_biaffine_model(input1, input2)
                            right_logits = right_biaffine_model(input1, input2)
                            loss = (crit(left_logits.squeeze(),left_labels.squeeze().double()) * mask).sum() + (crit(right_logits.squeeze(),right_labels.squeeze().double()) * mask).sum()
                            biaffine_loss.append(loss / 2) 

            # if args.return_h:
            #     raw_loss, hidden = ret
            #     train_loss += raw_loss.sum().item()
            #     loss = raw_loss.mean()
            #     loss = loss + args.alpha * hidden.pow(2).mean()
            #     loss = loss + args.beta * ((hidden[1:] - hidden[:-1]).pow(2)).mean()
            # else:
            # print(f"Biaffine forwarding takes {time.time()-biaffine_start:.2f} seconds")
            raw_loss = ret
            if not args.stage_two or (args.stage_two and epoch == args.num_epochs - 1): # change
                raw_biaffine_loss = torch.stack(biaffine_loss)
                # loss = 1 * (1/(1+args.BTloss_ratio) * raw_loss.mean() + args.BTloss_ratio/(1+args.BTloss_ratio) * raw_biaffine_loss.mean())
                loss = 1 * (1/(1+BTloss_ratio_list[train_step]) * raw_loss.mean() + BTloss_ratio_list[train_step]/(1+BTloss_ratio_list[train_step]) * raw_biaffine_loss.mean())
                train_biaffine_loss += raw_biaffine_loss.sum().item()
            else:
                loss = raw_loss.mean()
            train_loss += raw_loss.sum().item()

            loss.backward()
           
            # total_norm = 0.0
            # for p in left_biaffine_model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.detach().data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            
            # logger.info(f"Step {train_step + 1} left_biaffine_model Gradient Norm: {total_norm:.4f}")

            # total_norm = 0.0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.detach().data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            
            # logger.info(f"Step {train_step + 1} Transformer Gradient Norm: {total_norm:.4f}")
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(left_biaffine_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(right_biaffine_model.parameters(), args.max_grad_norm)
            optimizer.step()
            left_biaffine_optimizer.step()
            right_biaffine_optimizer.step()
            train_step += 1
            if args.scheduler == 'const':
                pass
            elif train_step < warm_up_step:
                warm_up_scheduler.step()
                left_biaffine_warm_up_scheduler.step()
                right_biaffine_warm_up_scheduler.step()
            elif args.scheduler == 'cosine':
                if train_step < decay_steps:
                    scheduler.step()
                    left_biaffine_scheduler.step()
                    right_biaffine_scheduler.step()
                else:
                    for j in range(len(optimizer.param_groups)):
                        optimizer.param_groups[j]['lr'] = args.stable_lr
                    for j in range(len(left_biaffine_optimizer.param_groups)):
                        left_biaffine_optimizer.param_groups[j]['lr'] = args.stable_lr
                    for j in range(len(right_biaffine_optimizer.param_groups)):
                        right_biaffine_optimizer.param_groups[j]['lr'] = args.stable_lr

            num_words += total_length
            num_sents += batch_size

            if train_step % args.log_every == 0:    #, B lr {biaffine_optimizer.param_groups[0]['lr']:.6f}
                logger.info(f"train step {train_step}, T lr {optimizer.param_groups[0]['lr']:.6f}, B lr {left_biaffine_optimizer.param_groups[0]['lr']:.6f}, T loss {train_loss / num_words:.4f}, B loss {train_biaffine_loss / num_words:.4f}")
                if (train_loss + train_biaffine_loss) / num_words <= 30:
                    logger.info(f"token ppl {np.exp(train_loss / num_words):.4f}, total ppl {np.exp((train_loss + train_biaffine_loss) / num_words):.4f}")
                num_words = 0
                num_sents = 0
                train_loss = 0.0  
                train_biaffine_loss = 0.0

                # logger.info(f"dev data evaluation ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
            
            if train_step % args.eval_interval == 0 or i == len(train_data) - 1:
                val_ppl, val_f1 = eval(dev_data, dev_index_to_id, left_dev_arrow, right_dev_arrow, startofword_dev, model,
                    left_biaffine_model, right_biaffine_model, dev_length, left_arc, right_arc, args=args)

                # if epoch == args.num_epochs - 2 and i == len(train_data) - 1:
                #     checkpoint = {'args': args,
                #                 'model': model.cpu(),
                #                 'left_biaffine_model': left_biaffine_model.cpu(),
                #                 'right_biaffine_model': right_biaffine_model.cpu(),
                #                 'vocab': vocab,
                #                 'optimizer': optimizer.state_dict(),
                #                 'left_biaffine_optimizer': left_biaffine_optimizer.state_dict(),
                #                 'right_biaffine_optimizer': right_biaffine_optimizer.state_dict(),
                #                 'scheduler': scheduler.state_dict() if args.scheduler == 'cosine' else None,
                #                 'left_biaffine_warm_up_scheduler': left_biaffine_warm_up_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                #                 'right_biaffine_warm_up_scheduler': right_biaffine_warm_up_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                #                 'warm_up_scheduler': warm_up_scheduler.state_dict(),
                #                 'left_biaffine_scheduler': left_biaffine_scheduler.state_dict(),
                #                 'right_biaffine_scheduler': right_biaffine_scheduler.state_dict(),
                #                 }
                #     torch.save(checkpoint, "./models/pretrain_for_biaffine_3_distance.pt")

                if val_ppl < best_val_ppl:
                    remaining_epoch = 0
                    best_val_ppl = val_ppl
                    best_val_f1 = val_f1
                    logger.info(f"new best ppl {best_val_ppl:.4f}, f1 {best_val_f1:.4f}")
                    checkpoint = {'args': args,
                                'model': model.cpu(),
                                'left_biaffine_model': left_biaffine_model.cpu(),
                                'right_biaffine_model': right_biaffine_model.cpu(),
                                'vocab': vocab,
                                'optimizer': optimizer.state_dict(),
                                'left_biaffine_optimizer': left_biaffine_optimizer.state_dict(),
                                'right_biaffine_optimizer': right_biaffine_optimizer.state_dict(),
                                'scheduler': scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'left_biaffine_warm_up_scheduler': left_biaffine_warm_up_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'right_biaffine_warm_up_scheduler': right_biaffine_warm_up_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'warm_up_scheduler': warm_up_scheduler.state_dict(),
                                'left_biaffine_scheduler': left_biaffine_scheduler.state_dict(),
                                'right_biaffine_scheduler': right_biaffine_scheduler.state_dict(),
                                }
                    torch.save(checkpoint, args.save_path)
                    if torch.cuda.is_available():
                        model.cuda()
                        left_biaffine_model.cuda()
                        right_biaffine_model.cuda()
                    test_ppl, test_f1 = eval(test_data, test_index_to_id, left_test_arrow, right_test_arrow, startofword_test, model, 
                        left_biaffine_model, right_biaffine_model, test_length, left_arc, right_arc, args=args)
                    logger.info(f"test ppl {test_ppl:.4f}, f1 {test_f1:.4f}")
                elif args.scheduler == 'decay':
                    remaining_epoch += 1
                    if remaining_epoch >= args.decay_interval:
                        remaining_epoch = 0
                        for j in range(len(optimizer.param_groups)):
                            optimizer.param_groups[j]['lr'] = max(optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        for j in range(len(left_biaffine_optimizer.param_groups)):
                            left_biaffine_optimizer.param_groups[j]['lr'] = max(left_biaffine_optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        for j in range(len(right_biaffine_optimizer.param_groups)):
                            right_biaffine_optimizer.param_groups[j]['lr'] = max(right_biaffine_optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        logger.info(f"decay lr to {optimizer.param_groups[0]['lr']:.6f}")
    
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