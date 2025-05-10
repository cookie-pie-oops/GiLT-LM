from datasets import Dataset as HFDataset
from model_bllip_con import PushdownTransformerConstituency
import torch
import numpy as np
import sentencepiece as spm
from config import ModelConfig, TrainConfig, ParallelConfig
import logging
import torch.nn as nn
import json
import gc
import random, os
from collate import collate_fn
DEBUG = False
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="push_bllip_con_test_gas1")
script_args = parser.parse_args()

@torch.inference_mode()
def eval_joint(model_args: ModelConfig, train_args: TrainConfig, parallel_args: ParallelConfig):
    # use the model to test
    
    
    test_ds = HFDataset.load_from_disk(
        "./data/BLLIP_LG_test"
    )
    if DEBUG:
        test_ds = test_ds.select(range(10))
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=train_args.batch_size if not DEBUG else 1,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )
    concatenated_attachment_labels = []
    for i in range(len(test_ds)):
        concatenated_attachment_labels += test_ds[i]["attachment_labels"]
    real_max_stack_depth = max(concatenated_attachment_labels)

    sp = spm.SentencePieceProcessor()
    sp.Load("./data_process/spm_parsing/BLLIP_spm.model")
    vocab_size = sp.GetPieceSize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # shut down the grad
    torch.set_grad_enabled(False)
    # test
    logging.info("=" * 100)
    logging.info(f"Testing started")
    # load best model
    model = PushdownTransformerConstituency(
        vocab_size=vocab_size,
        w_dim=model_args.w_dim,
        n_head=model_args.n_head,
        d_head=model_args.d_head,
        d_inner=model_args.d_inner,
        num_layers=model_args.num_layers,
        dropout=model_args.dropout,
        dropoutatt=model_args.dropoutatt,
        pad_id=model_args.pad_id,
        bos_id=model_args.bos_id,
        eos_id=model_args.eos_id,
        stack_pad_id=model_args.stack_pad_id,
        pre_lnorm=model_args.pre_lnorm,
        max_stack_depth=model_args.max_stack_depth
    ).to(device)
    # data parallel?
    if parallel_args.parallel == 'dp':
        model = torch.nn.DataParallel(model)

    ckpt = f"./ckpt/{script_args.run_name}/best/model.pt"
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(model, torch.nn.DataParallel):
        # if module. not in state_dict.keys():
        first_key = next(iter(state_dict.keys()))
        if "module." not in first_key:
            # 说明是单卡训练的模型
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict["module." + k] = v
            state_dict = new_state_dict
    else:
        # delete module.
        first_key = next(iter(state_dict.keys()))
        if "module." in first_key:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # eval the parallel module
    if parallel_args.parallel == "dp" and torch.cuda.device_count() > 1:
        model.module.eval()
    logging.info(f"Model loaded from {ckpt}")
    logging.info(f"Max stack depth: {real_max_stack_depth}")
    with torch.no_grad():
        test_losses_w = []
        test_losses_a = []
        # tp[i] = tps of i-th attach label
        tp = [0] * (real_max_stack_depth + 1)
        pre = [0] * (real_max_stack_depth + 1)
        tru = [0] * (real_max_stack_depth + 1)
        for _, test_batch in enumerate(test_dataloader):
            ids = test_batch["ids"]
            ids = torch.cat([torch.full((ids.shape[0], 1), model_args.bos_id), ids], dim=1)
            target = ids[:, 1:].to(device)
            data = ids[:, :-1].to(device)
            stack_tape = test_batch["stack_tape"].to(device)
            attachment_labels = test_batch["attachment_labels"].to(device)
            # forward
            loss_w, loss_a, attachment_preds = model.forward(
                data,
                target,
                stack_tape,
                attachment_labels,
                return_decoded_attach=True,
            )
            # f1
            attachment_preds = attachment_preds.view(-1) # [B,T] -> [B*T]
            attachment_labels = attachment_labels.view(-1) # [B*T]
            for idx in range(attachment_labels.shape[0]):
                if attachment_labels[idx] != model_args.stack_pad_id:
                    tru[attachment_labels[idx]] += 1
                    pre[attachment_preds[idx]] += 1
                    if attachment_labels[idx] == attachment_preds[idx]:
                        tp[attachment_labels[idx]] += 1
                         
            # data, target, stack_tape, attachment_labels = data.detach().cpu(), target.detach().cpu(), stack_tape.detach().cpu(), attachment_labels.detach().cpu()
            loss_w = loss_w.detach().cpu().sum()
            loss_a = loss_a.detach().cpu().sum()
            test_losses_w.append(loss_w.item())
            test_losses_a.append(loss_a.item())
            # if DEBUG:
            #     one_ppl = torch.exp(torch.tensor((loss_w + loss_a) / (ids.shape[1] - 1)))
            #     logging.info(f"Single sentence PPL: {one_ppl.item()}")
            # breakpoint()
        test_loss_w_sum = np.sum(test_losses_w)
        test_loss_a_sum = np.sum(test_losses_a)
        n_words = np.sum([torch.sum(test_batch["lengths"]).item() for test_batch in test_dataloader])
        test_loss_w = test_loss_w_sum / n_words
        test_loss_a = test_loss_a_sum / n_words
        test_loss = test_loss_w + test_loss_a
        word_ppl = torch.exp(torch.tensor(test_loss_w))
        all_ppl = torch.exp(torch.tensor(test_loss))
        
        # f1
        tp = np.array(tp)
        pre = np.array(pre)
        tru = np.array(tru)
        precision = tp / (pre + 1e-10) # prec of each label
        recall = tp / (tru + 1e-10) # recall of each label
        macro_f1 = 2 * precision * recall / (precision + recall + 1e-10)
        macro_f1 = np.mean(macro_f1)
        # micro f1
        micro_prec = np.sum(tp) / (np.sum(pre) + 1e-10)
        micro_rec = np.sum(tp) / (np.sum(tru) + 1e-10)
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-10)
        # print(precision, recall)
        logging.info(f"[Joint] Test Loss: {test_loss}, Word PPL: {word_ppl.item()}, # of tokens: {n_words}, Joint PPL: {all_ppl.item()}")
        logging.info(f"[Joint] Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        # write a file about test loss and step (json)
        with open(f"./ckpt/{script_args.run_name}/best/test_joint.json", "w") as f:
            ddict = {
                "test_joint_loss": test_loss,
                "test_word_loss": test_loss_w,
                "test_attachment_loss": test_loss_a,
                "test_joint_ppl": all_ppl.item(),
                "test_word_ppl": word_ppl.item(),
                "test_macro_f1": macro_f1,
                "test_micro_f1": micro_f1,
            }
            json.dump(ddict, f)
        logging.info(f"Test finished")
        logging.info("=" * 100)

if __name__ == "__main__":
    # load config
    model_args = ModelConfig()
    train_args = TrainConfig()
    parallel_args = ParallelConfig()
    # set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"./ckpt/{script_args.run_name}/best/test_joint.log"),
            logging.StreamHandler()
        ]
    )
    # eval
    eval_joint(model_args, train_args, parallel_args)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   