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
parser.add_argument('--TBloss_ratio', default=1.0, type=float)


def log_arguments(args):

    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")

def load_data(path, batchsize=-1, shuffle=False, seed=1111):

    with open(path, 'r') as f:
        # sents = [line.strip() for line in f.readlines()][:1000]
        
        # sents = [line.strip() for line in f.readlines()]
        # sents = sents[:len(sents)//4]
        
        sents = [line.strip() for line in f.readlines()]
        sents = [sent.split(',') if len(sent)!=0 else [] for sent in sents]
        sents = [[int(word) for word in sent] if sent != [] else [] for sent in sents]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]


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
                if word_id in [left_arc, right_arc, left_arc2, right_arc2, bos_id, eos_id]:
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
    arc_pos = idx
    while idx > 0:
        if index_to_id[idx] < 0:
            idx -= 1
        else:
            break
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
    return (span_hidden + hidden[arc_pos])/(span_len + 1) , position 
    #span_hidden/span_len/2 + hidden[arc_pos]/2  (span_hidden + hidden[arc_pos])/(span_len + 1)

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

def eval(data, index_to_id, arrow,startofword, model, biaffine_model, length, left_arc, right_arc, args = None):
    model.eval()
    biaffine_model.eval()
    num_sents = 0
    total_loss = 0.0
    num_words = 0
    acc = 0
    num_arcs = 0
    with torch.no_grad():
        for i in range(len(data)):
            sents = data[i]
            sents_index_to_id = index_to_id[i]
            sents_arrow = arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            mems = tuple()
            
            ret, hidden = model(sents, startofword[i], length[i], args.attn_mask, args.document_level, args.return_h, False, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id, sents_arrow=sents_arrow)
            hidden = hidden.transpose(0, 1)
            batch_words_input = hidden_alignment(hidden, sents_index_to_id)

            if sents_arrow:
                for sent_ids, sent_hidden, sent_index_to_id, words_input, sent_label in zip(
                    sents, hidden, sents_index_to_id, batch_words_input, sents_arrow):
                    for j in range(len(sent_ids)):
                        if sent_ids[j] == left_arc or sent_ids[j] == right_arc:
                            predicate_input, words_index = get_span(sent_hidden, sent_index_to_id, j)
                            words_piece_input = torch.stack(words_input[:words_index - 1])
                            logits = biaffine_model(predicate_input.view(1, 1, -1), words_piece_input.view(1, -1, args.w_dim))
                            # pred = (logits.view(1, -1) > 0.5)
                            pred = torch.argmax(logits, dim=2).item()
                            label = sent_label[j] - 1
                            # if pred[0][label] == True:
                            if pred == label:
                                acc += 1
                            num_arcs += 1

            num_words += total_length
            num_sents += batch_size
            total_loss += ret.sum().item()

    ppl = np.exp(total_loss / num_words) 
    logger = get_logger()
    logger.info(f"eval ppl {ppl:.4f}, acc {acc/num_arcs:.4f}")
    model.train()
    biaffine_model.train()
    return ppl, acc / num_arcs


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = args.train_file
    dev_path = args.dev_file
    test_path = args.test_file
    train_arrow_path = args.train_arrow_file
    dev_arrow_path = args.dev_arrow_file
    test_arrow_path = args.test_arrow_file

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size

    train_data = load_data(train_path, batchsize=batch_size, shuffle=True, seed=args.seed)
    dev_data = load_data(dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    test_data = load_data(test_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)

    train_arrow = load_data(train_arrow_path, batchsize=batch_size, shuffle=True, seed=args.seed)
    dev_arrow = load_data(dev_arrow_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    test_arrow = load_data(test_arrow_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)

    vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id, vocab = load_vocab(args.vocab_file)
    # print(left_arc)
    # print(right_arc)
    # print(len(startofword_id))
    train_data, startofword_train, train_length, train_index_to_id = add_to_all(train_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    dev_data, startofword_dev, dev_length, dev_index_to_id = add_to_all(dev_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)
    test_data, startofword_test, test_length, test_index_to_id = add_to_all(test_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, left_arc2, right_arc2, startofword_id)

    train_arrow = align_data_arrow(train_arrow, train_data, left_arc, right_arc)
    dev_arrow = align_data_arrow(dev_arrow, dev_data, left_arc, right_arc)
    test_arrow = align_data_arrow(test_arrow, test_data, left_arc, right_arc)

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
        biaffine_model = BiaffineAttention(args.w_dim, args.proj_dim)
        logger.info(f"Transformer parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in biaffine_model.parameters())}")
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
        if 'biaffine_model' in checkpoint:
            biaffine_model = checkpoint['biaffine_model']
            logger.info("Load exist biaffine model")
        else:
            biaffine_model = BiaffineAttention(args.w_dim, args.proj_dim)
        model = checkpoint['model']
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"biaffine model parameter counts: {sum(p.numel() for p in biaffine_model.parameters())}")
    
    nonemb_params = [p for p in model.parameters() if p.size() != (vocab_size, args.w_dim)]
    emb_params = list(model.emb.parameters())
    # param_list = [biaffine_model.parameters(), nonemb_params, emb_params]
    # lr_list = [1, 1, args.emb_lr_multiplier]
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
    
    # crit = nn.BCELoss(reduction="mean")
    crit = nn.NLLLoss(reduction="mean")

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
    
    # if args.model_file != '':
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     warm_up_scheduler.load_state_dict(checkpoint['warm_up_scheduler'])
    #     if torch.cuda.is_available():
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if torch.is_tensor(v):
    #                     state[k] = v.cuda()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()
        biaffine_model.cuda()
    model.train()
    biaffine_model.train()

    best_val_ppl = 1e5
    best_val_acc = 0
    train_step = 0
    remaining_epoch = 0
    
    checkpoint_step = 0
    for epoch in range(args.num_epochs):
        logger.info(f"epoch {epoch+1}")
        num_words = 0
        num_sents = 0
        train_loss = 0.0
        train_biaffine_loss = 0.0
        for i in range(len(train_data)):
            if train_step <= checkpoint_step:
                checkpoint_step -= 1
                # train_step += 1

                # optimizer.zero_grad()
                # biaffine_optimizer.zero_grad()
                # optimizer.step()
                # biaffine_optimizer.step()
                # if args.scheduler == 'const':
                #     pass
                # elif train_step < warm_up_step:
                #     warm_up_scheduler.step()
                #     biaffine_warm_up_scheduler.step()
                # elif args.scheduler == 'cosine':
                #     if train_step < decay_steps:
                #         scheduler.step()
                #         biaffine_scheduler.step()
                #     else:
                #         for i in range(len(optimizer.param_groups)):
                #             optimizer.param_groups[i]['lr'] = args.stable_lr
                #         for i in range(len(biaffine_optimizer.param_groups)):
                #             biaffine_optimizer.param_groups[i]['lr'] = args.stable_lr
                continue
            torch.cuda.empty_cache()
            tmp_time = time.time()
            sents = train_data[i]
            sents_index_to_id = train_index_to_id[i]
            sents_arrow = train_arrow[i]

            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            optimizer.zero_grad()
            biaffine_optimizer.zero_grad()
            mems = tuple()
            model : TransformerGrammar

            # model.eval()
            # with torch.no_grad():
            #     ret, hidden = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
            #             args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id, sents_arrow=sents_arrow)
            ret, hidden = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
                        args.max_relative_length, args.min_relative_length, sents_index_to_id=sents_index_to_id, sents_arrow=sents_arrow)

            hidden = hidden.transpose(0, 1)
            batch_words_input = hidden_alignment(hidden, sents_index_to_id)

            biaffine_start = time.time()
            # print(f"Transformers forwarding takes {biaffine_start-tmp_time:.2f} seconds")
            biaffine_loss = []
            max_length = max([len(sent) for sent in sents])
            if sents_arrow:
                for sent_ids, sent_hidden, sent_index_to_id, words_input, sent_label in zip(
                    sents, hidden, sents_index_to_id, batch_words_input, sents_arrow):
                    input1 = []
                    input2 = []
                    labels = []
                    for j in range(len(sent_ids)):
                        if sent_ids[j] == left_arc or sent_ids[j] == right_arc:
                            predicate_input, words_index = get_span(sent_hidden, sent_index_to_id, j)
                            padding_length = max_length - len(words_input[:words_index - 1])
                            if torch.cuda.is_available():
                                words_piece_input = torch.stack(words_input[:words_index - 1] + [torch.zeros(args.w_dim).cuda()]*padding_length)
                            else:
                                words_piece_input = torch.stack(words_input[:words_index - 1] + [torch.zeros(args.w_dim)]*padding_length) 
                            input1.append(predicate_input.view(1, -1))
                            input2.append(words_piece_input.view(-1, args.w_dim))
                            if torch.cuda.is_available():
                                label = torch.tensor([sent_label[j] - 1]).cuda()
                                # label = torch.zeros((max_length)).cuda()
                            else:
                                label = torch.tensor([sent_label[j] - 1])
                                # label = torch.zeros((max_length))
                            # label[sent_label[j] - 1] = 1
                            labels.append(label)
                    if input1:
                        input1 = torch.stack(input1)
                        input2 = torch.stack(input2)
                        labels = torch.stack(labels)
                        logits = biaffine_model(input1, input2)
                        loss = crit(torch.log(logits.transpose(1,2)),labels)
                        biaffine_loss.append(loss)
            
            # if args.return_h:
            #     raw_loss, hidden = ret
            #     train_loss += raw_loss.sum().item()
            #     loss = raw_loss.mean()
            #     loss = loss + args.alpha * hidden.pow(2).mean()
            #     loss = loss + args.beta * ((hidden[1:] - hidden[:-1]).pow(2)).mean()
            # else:
            # print(f"Biaffine forwarding takes {time.time()-biaffine_start:.2f} seconds")
            raw_loss = ret
            raw_biaffine_loss = torch.stack(biaffine_loss).mean()
            loss = 1/(1+args.TBloss_ratio) * raw_loss.mean() + args.TBloss_ratio/(1+args.TBloss_ratio) * raw_biaffine_loss
            train_loss += raw_loss.sum().item()
            
            train_biaffine_loss += raw_biaffine_loss.item()
            tmp_time2 = time.time()
            # print(f"forward time {tmp_time2 - tmp_time:.2f} s")

            loss.backward()
            tmp_time3 = time.time()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = args.stable_lr
                    for i in range(len(biaffine_optimizer.param_groups)):
                        biaffine_optimizer.param_groups[i]['lr'] = args.stable_lr
            # print(f"backward time {tmp_time3 - tmp_time2:.2f} s")
            num_words += total_length
            num_sents += batch_size

            if train_step % args.log_every == 0:    #, B lr {biaffine_optimizer.param_groups[0]['lr']:.6f}
                logger.info(f"train step {train_step}, T lr {optimizer.param_groups[0]['lr']:.6f}, B lr {biaffine_optimizer.param_groups[0]['lr']:.6f}, T loss {train_loss / num_words:.4f}, B loss {train_biaffine_loss:.4f}, ppl {np.exp(train_loss / num_words):.4f}")
                num_words = 0
                num_sents = 0
                train_loss = 0.0  
                train_biaffine_loss = 0.0

                # logger.info(f"dev data evaluation ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
            
            if train_step % args.eval_interval == 0 or i == len(train_data) - 1:
                val_ppl, val_acc = eval(dev_data, dev_index_to_id, dev_arrow, startofword_dev, model, biaffine_model, dev_length, left_arc, right_arc, args=args)

                if val_ppl < best_val_ppl:
                    remaining_epoch = 0
                    best_val_ppl = val_ppl
                    best_val_acc = val_acc
                    logger.info(f"new best ppl {best_val_ppl:.4f}, acc {best_val_acc:.4f}")
                    checkpoint = {'args': args,
                                'model': model.cpu(),
                                'biaffine_model': biaffine_model.cpu(),
                                'vocab': vocab,
                                'optimizer': optimizer.state_dict(),
                                'biaffine_optimizer': biaffine_optimizer.state_dict(),
                                'scheduler': scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'biaffine_warm_up_scheduler': biaffine_warm_up_scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'warm_up_scheduler': warm_up_scheduler.state_dict(),
                                'biaffine_scheduler': biaffine_scheduler.state_dict(),
                                }
                    torch.save(checkpoint, args.save_path)
                    if torch.cuda.is_available():
                        model.cuda()
                        biaffine_model.cuda()
                    test_ppl, test_acc = eval(test_data, test_index_to_id, test_arrow, startofword_test, model, biaffine_model, test_length, left_arc, right_arc, args=args)
                    logger.info(f"test ppl {test_ppl:.4f}, acc {test_acc:.4f}")
                elif args.scheduler == 'decay':
                    remaining_epoch += 1
                    if remaining_epoch >= args.decay_interval:
                        remaining_epoch = 0
                        for i in range(len(optimizer.param_groups)):
                            optimizer.param_groups[i]['lr'] = max(optimizer.param_groups[i]['lr'] * args.decay_rate, args.min_lr)
                        for i in range(len(biaffine_optimizer.param_groups)):
                            biaffine_optimizer.param_groups[i]['lr'] = max(biaffine_optimizer.param_groups[i]['lr'] * args.decay_rate, args.min_lr)
                        logger.info(f"decay lr to {optimizer.param_groups[0]['lr']:.6f}")
    
    end_time = time.time()
    logger.info(f"total time {end_time - start_time:.2f} s")
    logger.info(f"best val ppl {best_val_ppl:.4f}, acc {best_val_acc:.4f}")
    logger.info(f"best test ppl {test_ppl:.4f}, acc {test_acc:.4f}")
    logger.info(f"model saved to {args.save_path}")
    logger.info(f"log saved to {args.log_file}")
    logger.info(f"Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)