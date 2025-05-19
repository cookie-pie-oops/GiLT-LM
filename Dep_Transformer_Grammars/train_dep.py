import numpy as np
import torch
import argparse
import os
import sys
import csv
import torch.nn as nn
import logging
import time
import json
from scipy.stats import pearsonr, spearmanr
from torch import cuda
from helping_utils.logger import configure_logger, get_logger
from model_bllip_dep import TransformerGrammar, find_label_idx

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/train_LG_bllip_action.csv', type=str)
parser.add_argument('--dev_file', default='data/dev_bllip_action.csv', type=str)
parser.add_argument('--test_file', default='data/test_bllip_action.csv', type=str)
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
parser.add_argument('--finetune', default=None, type=str)
parser.add_argument('--sts_train_path', default=None)
parser.add_argument('--sts_dev_path', default=None)
parser.add_argument('--sts_test_path', default=None)
parser.add_argument('--write_test_output', default=None)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def log_arguments(args):

    logger = get_logger()
    hp_dict = vars(args)
    for key, value in hp_dict.items():
        logger.info(f"{key}\t{value}")

def load_data(path, batchsize=-1, shuffle=False, seed=1111):
    
    with open(path, 'r') as f:
        sents = [line.strip() for line in f.readlines()]
        # if "TRAIN" in path:
        #     sents = sents[:len(sents)//4]
        sents = [sent.split(',') for sent in sents]
        sents = [[int(word) for word in sent] for sent in sents]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]

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

def add_to_all(data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, startofword_id, pop_root):
    
    max_length = []
    startofword_copy = []
    for batch in data:
        max_tmp = 0
        batch_startofword = []
        for sent in batch:
            sent.insert(0, bos_id)
            sent.append(eos_id)
            arc_num = sum([1 for word in sent if word in [right_arc, left_arc]])
            sent_startofword = [vocab_size if startofword_id[word] == 1 else word for word in sent]
            batch_startofword.append(sent_startofword)
            length = len(sent) + arc_num
            if length > max_tmp:
                max_tmp = length
        startofword_copy.append(batch_startofword)
        max_length.append(max_tmp)
    
    return data, startofword_copy, max_length

def load_vocab(path):
    vocab_file = path

    pad_id = None
    bos_id = None
    eos_id = None
    left_arc = None
    right_arc = None
    pop_root = None

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
            elif vocab[i] == '->2' or vocab[i] == '<-2':
                pop_root = i
            elif vocab[i].startswith('▁'):
                startofword_id[i] = 1

    return vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, pop_root, startofword_id, vocab

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

def eval(data, startofword, model, length, args = None, score = None, write_test_output = None):
    model.eval()
    num_sents = 0
    total_loss = 0.0
    num_words = 0
    uas = 0
    microf1 = 0
    pre = 0
    infer = 0
    label = 0
    prediction = []
    score_label = []
    if write_test_output is not None:
        head_data = [["index", "prediction"]]
        count = 0
    with torch.no_grad():
        for i in range(len(data)):
            sents = data[i]
            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            mems = tuple()
            
            ret, prob = model(sents, startofword[i], length[i], args.attn_mask, args.document_level, False, 
                        args.max_relative_length, args.min_relative_length, finetune=args.finetune)
            if args.finetune == "sst2":
                # 26858,5599 -> 2056: positive 2060: negative
                out = np.argmax(prob.cpu(), axis=1)
                for j in range(len(sents)):
                    idx = find_label_idx(sents[j], [26858, 5599]) + 1
                    if sents[j][idx + 1] == 2060:   # 0
                        if prob[idx][2056][j] < prob[idx][2060][j]:
                            microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                    elif sents[j][idx + 1] == 2056:  # 1
                        if prob[idx][2056][j] > prob[idx][2060][j]:
                            microf1 += 1
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
                            microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, 0])
                                count += 1
                        else:
                            infer += 1
                            if write_test_output is not None:
                                head_data.append([count, 1])
                                count += 1
                    elif sents[j][idx + 1] == 2064:  # 1
                        label += 1
                        if prob[idx][2064][j] > prob[idx][72][j]:
                        # if sents[j][idx] == out[idx-1][j].item():
                            pre += 1
                            infer += 1
                            microf1 += 1
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
                            microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, "not_entailment"])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, "entailment"])
                                count += 1
                    elif sents[j][idx + 1] == 221:  # 1
                        if prob[idx][221][j] > prob[idx][60][j]:
                            microf1 += 1
                            if write_test_output is not None:
                                head_data.append([count, "entailment"])
                                count += 1
                        else:
                            if write_test_output is not None:
                                head_data.append([count, "not_entailment"])
                                count += 1

            if args.finetune == "sts":
                out = ret.cpu().tolist()
                prediction.extend(out)
                score_label.extend(score[i])
                if write_test_output is not None:
                    for idx in range(len(sents)):
                        head_data.append([count, f"{out[idx]:.3f}"])
                        count += 1

            num_words += total_length
            num_sents += batch_size
            total_loss += ret.sum().item()

    if write_test_output is not None:
        fw = open(write_test_output, 'w')
        writer = csv.writer(fw, delimiter="\t")
        writer.writerows(head_data)
    logger = get_logger()
    model.train()
    if args.finetune is None:
        ppl = np.exp(total_loss / num_words) 
        logger.info(f"---eval ppl {ppl:.4f}")
        return ppl, uas
    elif args.finetune == "sst2" or args.finetune == "rte":
        microf1 = microf1 / num_sents
        logger.info(f"---eval f1 {microf1:.4f}")
        return -microf1, uas
    elif args.finetune == "mrpc":
        precision = pre / infer
        recall = pre / label
        logger.info(f"---acc {microf1 / num_sents:.4f}")
        microf1 = 2 * (precision*recall)/(precision+recall) if precision != 0 and recall != 0 else 0
        logger.info(f"---eval f1 {microf1:.4f}")
        return -microf1, uas
    elif args.finetune == "sts":
        prediction = np.array(prediction).reshape(-1)
        score_label = np.array(score_label).reshape(-1)
        r, _ = pearsonr(prediction, score_label)
        spr, _ = spearmanr(prediction, score_label)
        logger.info(f"---pearson correlation coefficient:{r:.4f}")
        logger.info(f"---spearman rank correlation coefficient:{spr:.4f}")
        return -(r+spr), uas

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = args.train_file
    dev_path = args.dev_file
    test_path = args.test_file
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size

    train_data = load_data(train_path, batchsize=batch_size, shuffle=True, seed=args.seed)
    dev_data = load_data(dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    test_data = load_data(test_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    if args.finetune == "sts":
        train_sts_score = load_STS_score(args.sts_train_path, batchsize=batch_size, shuffle=True, seed=args.seed)
        dev_sts_score = load_STS_score(args.sts_dev_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
        test_sts_score = load_STS_score(args.sts_test_path, batchsize=eval_batch_size, shuffle=True, seed=args.seed)
    vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, pop_root, startofword_id, vocab = load_vocab(args.vocab_file)
    
    # print(left_arc)
    # print(right_arc)
    # print(len(startofword_id))

    train_data, startofword_train, train_length = add_to_all(train_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, startofword_id, pop_root)
    dev_data, startofword_dev, dev_length = add_to_all(dev_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, startofword_id, pop_root)
    test_data, startofword_test, test_length = add_to_all(test_data, vocab_size, pad_id, bos_id, eos_id, left_arc, right_arc, startofword_id, pop_root)

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
                                   eos_id, left_arc, right_arc, pop_root, startofword_id, args.pre_lnorm)
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
        model.apply(weights_init)
        fan_in = nn.init._calculate_correct_fan(model.emb.weight, 'fan_in')
        logger.info(f"fan in {fan_in}")
        nn.init.uniform_(model.emb.weight, -np.sqrt(3 / fan_in), np.sqrt(3 / fan_in))
    else:
        logger.info(f"loading model from {args.model_file}")
        checkpoint = torch.load(args.model_file, map_location=torch.device(device), weights_only=False)
        model = checkpoint['model']
        logger.info(f"model parameter counts: {sum(p.numel() for p in model.parameters())}")
    
    nonemb_params = [p for p in model.parameters() if p.size() != (vocab_size, args.w_dim)]
    emb_params = list(model.emb.parameters())
    param_list = [nonemb_params, emb_params]
    lr_list = [1, args.emb_lr_multiplier]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params': p, 'lr': lr} for p, lr in zip(param_list, lr_list)], weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    total_steps = len(train_data) * args.num_epochs
    decay_steps = len(train_data) * args.decay_epochs
    warm_up_step = args.lr_warm_step
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps - warm_up_step, eta_min=args.eta_min)
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    elif args.scheduler == 'decay':
        warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (step / warm_up_step * (args.max_lr - args.start_lr) + args.start_lr) if step < warm_up_step else args.max_lr, last_epoch=-1)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.max_lr
    
    if args.model_file != '':
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        warm_up_scheduler.load_state_dict(checkpoint['warm_up_scheduler'])
    
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.finetune == "sts":
        model.STS = nn.Linear(args.w_dim, 1)
        nn.init.xavier_uniform_(model.STS.weight)
        nn.init.zeros_(model.STS.bias)
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

    model.to(device)
    model.train()

    best_val_ppl = 1e5
    best_val_uas = 0
    train_step = 0
    remaining_epoch = 0
    
    checkpoint_step = 0
    mseloss = nn.MSELoss(reduction='none')
    for epoch in range(args.num_epochs):
        logger.info(f"epoch {epoch + 1}")
        num_words = 0
        num_sents = 0
        train_loss = 0.0
        for i in range(len(train_data)):
            if train_step <= checkpoint_step:
                train_step += 1
                continue
            torch.cuda.empty_cache()
            tmp_time = time.time()
            sents = train_data[i]
            batch_size = len(sents)
            total_length = sum([len(sent) - 1 for sent in sents])
            optimizer.zero_grad()
            mems = tuple()
            model : TransformerGrammar
            # print(startofword_train[i])
            ret, _ = model(sents, startofword_train[i], train_length[i], args.attn_mask, args.document_level, args.return_h, 
                        args.max_relative_length, args.min_relative_length, finetune=args.finetune)
            if args.finetune == "sts":
                sts_scores = ret
                sts_label = torch.tensor(train_sts_score[i]).unsqueeze(1).to(device)
                ret = mseloss(sts_scores, sts_label)
            
            if args.return_h:
                raw_loss, hidden = ret
                train_loss += raw_loss.sum().item()
                loss = raw_loss.mean()
                loss = loss + args.alpha * hidden.pow(2).mean()
                loss = loss + args.beta * ((hidden[1:] - hidden[:-1]).pow(2)).mean()
            else:
                raw_loss = ret
                loss = raw_loss.mean()
                train_loss += raw_loss.sum().item()
            tmp_time2 = time.time()
            # print(f"forward time {tmp_time2 - tmp_time:.2f} s")
            loss.backward()
            tmp_time3 = time.time()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            train_step += 1
            
            if args.scheduler == 'const':
                pass
            elif train_step < warm_up_step:
                warm_up_scheduler.step()
            elif args.scheduler == 'cosine':
                if train_step < decay_steps:
                    scheduler.step()
                else:
                    for j in range(len(optimizer.param_groups)):
                        optimizer.param_groups[j]['lr'] = args.stable_lr
            # print(f"backward time {tmp_time3 - tmp_time2:.2f} s")
            num_words += total_length
            num_sents += batch_size

            if train_step % args.log_every == 0:
                logger.info(f"train step {train_step}, lr {optimizer.param_groups[0]['lr']:.6f}, loss {train_loss / num_words:.4f}, ppl {np.exp(train_loss / num_words):.4f}")
                num_words = 0
                num_sents = 0
                train_loss = 0.0  

                logger.info(f"dev data evaluation ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
            
            if train_step % args.eval_interval == 0 or i == len(train_data)-1:
                if args.finetune != "sts":
                    val_ppl, val_uas = eval(dev_data, startofword_dev, model, dev_length, args=args)
                else:
                    val_ppl, val_uas = eval(dev_data, startofword_dev, model, dev_length, args=args, score=dev_sts_score)

                if val_ppl < best_val_ppl:
                    remaining_epoch = 0
                    best_val_ppl = val_ppl
                    best_val_uas = val_uas
                    logger.info(f"new best ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
                    checkpoint = {'args': args,
                                'model': model,
                                'vocab': vocab,
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict() if args.scheduler == 'cosine' else None,
                                'warm_up_scheduler': warm_up_scheduler.state_dict()
                                }
                    torch.save(checkpoint, args.save_path) 
                    # model.cuda()
                    if args.finetune != "sts":
                        test_ppl, test_uas = eval(test_data, startofword_test, model, test_length, args=args, write_test_output=args.write_test_output)
                    else:
                        test_ppl, test_uas = eval(test_data, startofword_test, model, test_length, args=args, score=test_sts_score, write_test_output=args.write_test_output)
                    logger.info(f"test ppl {test_ppl:.4f}, uas {test_uas:.4f}")
                elif args.scheduler == 'decay':
                    remaining_epoch += 1
                    if remaining_epoch >= args.decay_interval:
                        remaining_epoch = 0
                        for j in range(len(optimizer.param_groups)):
                            optimizer.param_groups[j]['lr'] = max(optimizer.param_groups[j]['lr'] * args.decay_rate, args.min_lr)
                        logger.info(f"decay lr to {optimizer.param_groups[0]['lr']:.6f}")
    
    end_time = time.time()
    logger.info(f"total time {end_time - start_time:.2f} s")
    logger.info(f"best val ppl {best_val_ppl:.4f}, uas {best_val_uas:.4f}")
    if args.finetune != "sts" and args.finetune != "rte":
        logger.info(f"best test ppl {test_ppl:.4f}, uas {test_uas:.4f}")
    logger.info(f"model saved to {args.save_path}")
    logger.info(f"log saved to {args.log_file}")
    logger.info(f"Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)