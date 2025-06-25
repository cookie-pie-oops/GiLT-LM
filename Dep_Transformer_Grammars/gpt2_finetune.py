import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel #4.44.1
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from model_bllip_dep import BiaffineAttention
from train_graphLayer import predicate_alignment, load_STS_score
import os
import json
import csv
from scipy.stats import pearsonr, spearmanr

from helping_utils.logger import configure_logger, get_logger
logger = get_logger()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

metric_dict = {"MRPC": ["acc", "f1"], "SST2": ["acc"], "RTE": ["acc"], "STS": ["pearson", "spearman"]}

def load_data(path, batchsize=-1, shuffle=False, seed=1111, size="default"):

    with open(path, 'r') as f:
        if size == "demo":
            sents = [line.strip() for line in f.readlines()][:1000]
        elif size == "small":
            sents = [line.strip() for line in f.readlines()]
            sents = sents[:len(sents)//4]
        else:
            sents = [line.strip() for line in f.readlines()]
        # sents = [sent.split(',') for sent in sents]
        # sents = [[int(word) for word in sent] for sent in sents]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sents)
    
    if batchsize == -1:
        return [sents]
    else:
        return [sents[i:i+batchsize] for i in range(0, len(sents), batchsize)]


def eval(model, eval_data, tokenizer, eos_string, downstreamtask, write_file=False, eval_score=None):
    if downstreamtask == "STS":
        STS_out = []
        STS_label = []
        assert eval_score is not None
    sign_dict = {"MRPC":{"1": 7548, "0": 11082}, "RTE":{"1": 352, "0": 657}, "SST2":{"1": 3967, "0": 4633}}
    write_file_sign_dict = {"RTE":{"1": "entailment", "0": "not_entailment"}, "SST2":{"1":1, "0":0}, "MRPC":{"1": 1, "0": 0}}
    model.eval()
    total_acc = 0.0
    total_num = 0
    total_infer = 0
    total_pre = 0
    total_label = 0
    head_data = [["index", "prediction"]]
    count = 0
    bos_eos_id = 50256
    for idx, eval_strings in enumerate(eval_data):
        eval_idxs = [[bos_eos_id] + tokenizer.encode(tokenizer.decode(tokenizer.encode(eval_string)).strip()) + [bos_eos_id] for eval_string in eval_strings]
        # eval_idxs = [tokenizer.encode(eos_string + eval_string + eos_string)[1:] for eval_string in eval_strings]
        max_len = max([len(eval_idx) for eval_idx in eval_idxs])
        eval_inps = torch.ones((len(eval_idxs), max_len - 1), dtype=torch.long) * 0
        eval_tgts = torch.ones((len(eval_idxs), max_len - 1), dtype=torch.long) * -1
        for i, eval_idx in enumerate(eval_idxs):
            eval_id = torch.tensor(eval_idx, dtype=torch.long)
            eval_inps[i, :len(eval_idx) - 1] = eval_id[:-1]
            eval_tgts[i, :len(eval_idx) - 1] = eval_id[1:]
        eval_inps = eval_inps.to(device)
        eval_tgts = eval_tgts.to(device)
        
        outputs = model(eval_inps, labels=eval_inps, output_hidden_states=True)
        if downstreamtask != "STS":
            index = [len(eval_idx) - 3 if eval_idx[len(eval_idx) - 3] == 25 else len(eval_idx) - 4 for eval_idx in eval_idxs]
            prediction = torch.argmax(outputs.logits, dim=-1)[torch.arange(len(eval_idxs)), index]
            for pred in prediction:
                if pred == sign_dict[downstreamtask]["1"]:
                    head_data.append([count, write_file_sign_dict[downstreamtask]["1"]])
                else:
                    head_data.append([count, write_file_sign_dict[downstreamtask]["0"]])
                count += 1
            label = eval_tgts[torch.arange(len(eval_idxs)), index]
            acc_num = (prediction == label).sum().item()
            total_num += len(eval_idxs)
            total_acc += acc_num
            if downstreamtask == "MRPC":
                total_pre += ((prediction == sign_dict[downstreamtask]["1"]) & (label == sign_dict[downstreamtask]["1"])).sum().item()
                total_infer += (prediction == sign_dict[downstreamtask]["1"]).sum().item()
                total_label += (label == sign_dict[downstreamtask]["1"]).sum().item()
        else:
            prediction = model.STS(outputs.hidden_states[-1][:, -1, :])
            prediction = torch.clip(prediction, 0, 5).flatten().cpu().tolist()
            STS_out.extend(prediction)
            STS_label.extend(eval_score[idx])
            for score in prediction:
                head_data.append([count, float(score)])
                count += 1
    out_metric = {}
    for metric in metric_dict[downstreamtask]:
        if metric == "acc":
            out_metric[metric] = total_acc / total_num
        if metric == "f1":
            f1 = 2 * total_pre / (total_infer + total_label) if (total_infer + total_label) != 0 else 0
            out_metric[metric] = f1
    
    if downstreamtask == "STS":
        STS_out = np.array(STS_out).reshape(-1)
        STS_label = np.array(STS_label).reshape(-1)
        pearson, _ = pearsonr(STS_out, STS_label)
        spearman, _ = spearmanr(STS_out, STS_label)
        out_metric["pearson"] = pearson
        out_metric["spearman"] = spearman

    if write_file:
        fw = open(f"outputs/test_{downstreamtask}.tsv", 'w')
        writer = csv.writer(fw, delimiter="\t")
        writer.writerows(head_data)
    model.train()
    return out_metric

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

if __name__ == "__main__":
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    downstreamtask = "STS"
    train_bz_dict = {"SST2":64, "RTE":32, "STS":64, "MRPC":64}
    epoch_dict = {"SST2":5, "RTE":10, "STS":20, "MRPC":15}
    eval_interval_dict = {"SST2":100, "RTE":40, "STS":80, "MRPC":40}
    LR_dict = {"SST2":7.5e-6, "RTE":3.75e-6, "STS":7.5e-6, "MRPC":7.5e-6}   # SST2 3e-6
    train_path = f"/home/huangty/SDP_Transformer_project/data_process/{downstreamtask}/{downstreamtask}_TRAIN.txt"
    test_path = f"/home/huangty/SDP_Transformer_project/data_process/{downstreamtask}/{downstreamtask}_TEST.txt"
    dev_path = f"/home/huangty/SDP_Transformer_project/data_process/{downstreamtask}/{downstreamtask}_DEV.txt"

    train_data = load_data(train_path, batchsize=train_bz_dict[downstreamtask], seed=seed, shuffle=True)
    test_data = load_data(test_path, batchsize=8, seed=seed, shuffle=False)
    dev_data = load_data(dev_path, batchsize=8, seed=seed, shuffle=False)

    tokenizer = GPT2Tokenizer.from_pretrained("/home/huangty/GPT2/medium355M")
    tokenizer.add_prefix_space = True
    model = GPT2LMHeadModel.from_pretrained("/home/huangty/GPT2/medium355M") # output_hidden_states=True
    model.load_state_dict(torch.load('models/gpt2_medium_post.pt', map_location=device))
    eos_string = tokenizer.eos_token
    bos_string = tokenizer.bos_token
    bos_eos_id = 50256

    if downstreamtask == "STS":
        sts_train_path = "../data_process/STS/STS_TRAIN_score.txt"
        sts_dev_path = "../data_process/STS/STS_DEV_score.txt"
        sts_test_path = "../data_process/STS/STS_TEST_score.txt"
        train_sts_score = load_STS_score(sts_train_path, batchsize=train_bz_dict[downstreamtask], shuffle=True, seed=seed)
        dev_sts_score = load_STS_score(sts_dev_path, batchsize=8, shuffle=False, seed=seed)
        test_sts_score = load_STS_score(sts_test_path, batchsize=8, shuffle=False, seed=seed)
        model.STS = torch.nn.Linear(1024, 1)

    EPOCHS = epoch_dict[downstreamtask]
    LEARNING_RATE = LR_dict[downstreamtask]
    configure_logger(f"logs/gpt2_finetune_{downstreamtask}.log")
    logger = get_logger()

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    cs_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    mse_loss = torch.nn.MSELoss(reduction='none')
    logger.info(f"Task {downstreamtask}")
    best_metric = 0
    step_count = 0
    sum_loss = 0.0
    for epoch in range(EPOCHS):
        logger.info('=' * 30 + f"EPOCH {epoch + 1} started" + '=' * 30)
        for idx, train_strings in enumerate(train_data):
            train_idxs = [[bos_eos_id] + tokenizer.encode(tokenizer.decode(tokenizer.encode(train_string)).strip()) + [bos_eos_id] for train_string in train_strings]
            # train_idxs = [tokenizer.encode(eos_string + train_string + eos_string)[1:] for train_string in train_strings]
            max_len = max([len(train_idx) for train_idx in train_idxs])
            train_inps = torch.ones((len(train_idxs), max_len - 1), dtype=torch.long) * 0
            train_tgts = torch.ones((len(train_idxs), max_len - 1), dtype=torch.long) * -1
            for i, train_idx in enumerate(train_idxs):
                train_id = torch.tensor(train_idx, dtype=torch.long)
                train_inps[i, :len(train_idx) - 1] = train_id[:-1]
                train_tgts[i, :len(train_idx) - 1] = train_id[1:]
            train_inps = train_inps.to(device)
            train_tgts = train_tgts.to(device)
            # <bos> and <eos> are both 50256
                
            outputs = model(train_inps, labels=train_inps, output_hidden_states=True)
            if downstreamtask != "STS":
                loss = cs_loss(outputs.logits.permute(0,2,1), train_tgts)
                index = [len(train_idx) - 3 if train_idx[len(train_idx) - 3] == 25 else len(train_idx) - 4 for train_idx in train_idxs]
                loss = loss[torch.arange(len(train_idxs)), index].sum()
            else:
                loss = mse_loss(model.STS(outputs.hidden_states[-1][:, -1, :]), torch.tensor(train_sts_score[idx], device=device).unsqueeze(1)).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            # torch.nn.utils.clip_grad_norm_(biaffine_model.parameters(), 3.0)
            optimizer.step()
            sum_loss = sum_loss + loss.item()

            step_count += 1
            if step_count % 20 == 0:
                logger.info(f"Epoch {epoch+1} Step {step_count} / {len(train_data) * EPOCHS}, loss {sum_loss / 20:.4f}")
                sum_loss = 0.0

            if step_count % eval_interval_dict[downstreamtask] == 0 or step_count == len(train_data) * EPOCHS:
                # test on dev
                dev_metric_dict = eval(model, dev_data, tokenizer, eos_string, downstreamtask, eval_score=dev_sts_score)
                if downstreamtask == "MRPC":
                    dev_metric = dev_metric_dict["f1"]
                elif downstreamtask == "STS":
                    dev_metric = dev_metric_dict["pearson"] + dev_metric_dict["spearman"]
                else:
                    dev_metric = dev_metric_dict["acc"]
                for key, value in dev_metric_dict.items():
                    logger.info(f"Dev {key}: {value}")
                if dev_metric > best_metric:
                    best_metric = dev_metric
                    test_metric_dict = eval(model, test_data, tokenizer, eos_string, downstreamtask, True, eval_score=test_sts_score)
                    for key, value in test_metric_dict.items():
                        logger.info(f"Test {key}: {value}")
                    torch.save(model.state_dict(), os.path.join(f"models/gpt2_medium_{downstreamtask}.pt"))
    
    logger.info(f"Best metric: {best_metric}")
    for key, value in test_metric_dict.items():
        logger.info(f"Test {key}: {value}")
    logger.info(f"New model saved at models/gpt2_medium_{downstreamtask}.pt")
