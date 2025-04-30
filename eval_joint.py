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
from collate import collate_fn
DEBUG = False

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
        batch_size=1,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load("./data_process/spm_parsing/BLLIP_spm.model")
    vocab_size = sp.GetPieceSize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    ckpt = f"./ckpt/{train_args.run_name}/best/model.pt"
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
    
    with torch.no_grad():
        test_losses_w = []
        test_losses_a = []
        for _, test_batch in enumerate(test_dataloader):
            ids = test_batch["ids"]
            ids = torch.cat([torch.full((ids.shape[0], 1), model_args.bos_id), ids], dim=1)
            target = ids[:, 1:].to(device)
            data = ids[:, :-1].to(device)
            stack_tape = test_batch["stack_tape"].to(device)
            attachment_labels = test_batch["attachment_labels"].to(device)
            # forward
            loss_w, loss_a = model.forward(
                data,
                target,
                stack_tape,
                attachment_labels,
            )
            data, target, stack_tape, attachment_labels = data.detach().cpu(), target.detach().cpu(), stack_tape.detach().cpu(), attachment_labels.detach().cpu()
            loss_w = loss_w.detach().cpu().sum()
            loss_a = loss_a.detach().cpu().sum()
            test_losses_w.append(loss_w.item())
            test_losses_a.append(loss_a.item())
            # running_ppl = torch.exp(torch.tensor((loss_w + loss_a) / (ids.shape[1] - 1)))
            # breakpoint()
        test_loss_w_sum = np.sum(test_losses_w)
        test_loss_a_sum = np.sum(test_losses_a)
        n_words = np.sum([torch.sum(test_batch["lengths"]).item() for test_batch in test_dataloader])
        test_loss_w = test_loss_w_sum / n_words
        test_loss_a = test_loss_a_sum / n_words
        test_loss = test_loss_w + test_loss_a
        word_ppl = torch.exp(torch.tensor(test_loss_w))
        all_ppl = torch.exp(torch.tensor(test_loss))
        logging.info(f"[Joint] Test Loss: {test_loss}, Word PPL: {word_ppl.item()}, # of tokens: {n_words}, Joint PPL: {all_ppl.item()}")

        # write a file about test loss and step (json)
        with open(f"./ckpt/{train_args.run_name}/best/test.json", "w") as f:
            ddict = {
                "test_joint_loss": test_loss,
                "test_word_loss": test_loss_w,
                "test_attachment_loss": test_loss_a,
                "test_joint_ppl": all_ppl.item(),
                "test_word_ppl": word_ppl.item(),
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
            logging.FileHandler(f"./ckpt/{train_args.run_name}/best/test_joint.log"),
            logging.StreamHandler()
        ]
    )
    # eval
    eval_joint(model_args, train_args, parallel_args)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   