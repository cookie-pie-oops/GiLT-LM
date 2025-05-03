from datasets import Dataset as HFDataset
import argparse
import os
import random
from model_bllip_con import PushdownTransformerConstituency
import torch
import numpy as np
import sentencepiece as spm
from config import ModelConfig, TrainConfig, ParallelConfig
import logging
import torch.nn as nn
import json
import wandb
import datetime

parser = argparse.ArgumentParser()
# add: debug or not
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def main(model_args: ModelConfig, train_args: TrainConfig, parallel_args: ParallelConfig):
    if parallel_args.parallel == 'ddp':
        raise NotImplementedError("Distributed Data Parallel (DDP) is not implemented yet.")
    # 0. load datasets
    train_dataset = HFDataset.load_from_disk(
        "./data/BLLIP_LG_train"
    )
    dev_dataset = HFDataset.load_from_disk(
        "./data/BLLIP_LG_dev"
    )

    # 1. load model
    # 1.1 set seed
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    # torch.backends.cudnn.deterministic = True
    
    # 1.2 set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1.3 spm tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load("./data_process/spm_parsing/BLLIP_spm.model")
    
    vocab_size = sp.GetPieceSize()
    
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
    num_params_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    # model numel norm
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {num_params / 1e6:.2f}M parameters")
    # model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    logging.info(f"Model size: {model_size:.2f}MB")
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight'):
                scale = 2.0 / model_args.num_layers
                fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                nn.init.trunc_normal_(m.weight, 0.0, np.sqrt(scale / fan_in))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('PushdownTransformerConstituency') != -1:
            if hasattr(m, 'r_w_bias'):
                fan_in = nn.init._calculate_correct_fan(m.r_w_bias, 'fan_in')
                nn.init.trunc_normal_(m.r_w_bias, 0.0, np.sqrt(1.0 / fan_in))
            if hasattr(m, 'r_r_bias'):
                fan_in = nn.init._calculate_correct_fan(m.r_r_bias, 'fan_in')
                nn.init.trunc_normal_(m.r_r_bias, 0.0, np.sqrt(1.0 / fan_in))
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                nn.init.uniform_(m.weight, -np.sqrt(3.0 / fan_in), np.sqrt(3.0 / fan_in))
                
    model.apply(weights_init) # TODO: Maybe we don't need depth embedding. Watch the final PPL and adjust.

    if parallel_args.parallel == 'dp':
        model = torch.nn.DataParallel(model)
    
    # 2. load optimizer
    # 2.1 optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.max_lr,
        weight_decay=train_args.weight_decay,
    )
    # 2.2 scheduler
    # warmup + cosine annealing
    # use sequentialLR
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 
                                              start_factor=train_args.start_lr / train_args.max_lr,
                                              total_iters=train_args.warmup_steps // train_args.gradient_accumulation_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_args.epochs * len(train_dataset) // (train_args.batch_size * train_args.gradient_accumulation_steps)
                                                                            - train_args.warmup_steps // train_args.gradient_accumulation_steps, train_args.epochs),
                                                       eta_min=train_args.eta_min)
        ],
        milestones=[train_args.warmup_steps // train_args.gradient_accumulation_steps], # this is the step when the first scheduler will be used
    )
    # 3. train
    # 3.1 set model to train mode
    model.train()
    
    # 3.2 dataloader
    def collate_fn(batch):
        # find max len
        max_len = max(len(item["ids"]) for item in batch)
        
        padded_ids = []
        padded_stack_tape = []
        padded_attachment_labels = []
        lengths = []
        idxs = []
        
        pad_id = model_args.pad_id  # pad_id: 0 for spm vocab
        pad_value_stack = 0  # stack_tape 的 pad 值: max_depth - 1 (-100 will cause index error, and ?0 has been occupied?)
        pad_value_att = model_args.stack_pad_id    # attachment_labels 的 pad 值: -100 (0 has been occupied)
        
        # for item in batch:
        #     T = len(item["ids"])
        #     lengths.append(T)
        #     idxs.append(item.get("idx", 0))
            
        #     # 对 ids 进行 padding 到 max_len
        #     padded_ids.append(item["ids"] + [pad_id] * (max_len - T))
            
        #     # 对 stack_tape 进行 padding：原始是 T x T，需要 pad 到 max_len x max_len
        #     # 先初始化一个 max_len x max_len 的矩阵，用 pad_value_stack 填充
        #     padded_matrix = [[pad_value_stack] * max_len for _ in range(max_len)]
        #     for i in range(T):
        #         for j in range(T):
        #             padded_matrix[i][j] = item["stack_tape"][i][j]
        #     padded_stack_tape.append(padded_matrix)
            
        #     # 对 attachment_labels 进行 padding 到 max_len
        #     padded_attachment_labels.append(item["attachment_labels"] + [pad_value_att] * (max_len - T))
        
        for item in batch:
            T = len(item["ids"])
            lengths.append(T)
            idxs.append(item.get("idx", 0))
            
            # ids padding -> max_len
            padded_ids.append(item["ids"] + [pad_id] * (max_len - T))
            
            # pad stack_tape using tensor operations
            stack_tensor = torch.as_tensor(item["stack_tape"], dtype=torch.long)
            # full matrix with pad_value_stack
            padded_tensor = torch.full((max_len, max_len), pad_value_stack, dtype=torch.long)
            # fill the upper left corner with stack_tensor
            padded_tensor[:T, :T] = stack_tensor
            padded_stack_tape.append(padded_tensor) # so padded_stack_tape is a list of tensors
            
            # attachment_labels 的 padding
            padded_attachment_labels.append(item["attachment_labels"] + [pad_value_att] * (max_len - T))
        # to tensor
        
        batch_ids = torch.tensor(padded_ids, dtype=torch.long) # shape: [batch_size, max_len]
        
        batch_lengths = torch.tensor(lengths, dtype=torch.long) # shape: [batch_size]
        # batch_stack_tape = torch.tensor(padded_stack_tape, dtype=torch.long) # shape: [batch_size, max_len, max_len]
        batch_stack_tape = np.stack(padded_stack_tape, axis=0) # shape: [batch_size, max_len, max_len]
        batch_stack_tape = torch.as_tensor(batch_stack_tape, dtype=torch.long) # shape: [batch_size, max_len, max_len]
        batch_attachment_labels = torch.tensor(padded_attachment_labels, dtype=torch.long) # shape: [batch_size, max_len]
        batch_idxs = torch.tensor(idxs, dtype=torch.long) # shape: [batch_size]
        del padded_ids, padded_stack_tape, padded_attachment_labels, lengths, idxs
        
        return {
            "ids": batch_ids,
            "lengths": batch_lengths,
            "idxs": batch_idxs,
            "stack_tape": batch_stack_tape,
            "attachment_labels": batch_attachment_labels
        }
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )
    
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )
        
    # 3.3 training loop
    # table = wandb.Table(
    #     columns=["general_step", "text", "pieces", "ids", "attachment_labels"],
    # )
    best_eval_loss = float("inf")
    optimizer.zero_grad()
    for epoch in range(train_args.epochs):
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU,
        #                 torch.profiler.ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=2),
        #     record_shapes=True,
        #     profile_memory=True,
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #         "./tb/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))) as prof:
        # with nothing
        for _ in range(1):
            for step, batch in enumerate(train_dataloader):
                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                torch.cuda.empty_cache()
                general_step = epoch * len(train_dataloader) + step
                # 3.3.1 get data
                ids = batch["ids"] # [B, T]
                # cat bos before ids, where bos is model_args.bos_id
                ids = torch.cat([torch.full((ids.shape[0], 1), model_args.bos_id, device=device), ids.to(device)], dim=1) # [B, T+1]
                target = ids[:, 1:]
                data = ids[:, :-1]
                ids = ids.detach().cpu()
                # del ids

                # stack_tape is already padded in collate_fn
                stack_tape = batch["stack_tape"].to(device) # [B, T, T]
                attachment_labels = batch["attachment_labels"].to(device) # [B, T]
                
                # 3.3.2 forward
                loss_w, loss_a = model.forward(
                    data,
                    target,
                    stack_tape,
                    attachment_labels,
                ) # summed loss
                loss_w = loss_w.sum() # for dataparallel
                loss_a = loss_a.sum()
                loss = (loss_w + train_args.attachment_ratio * loss_a) / (1 + train_args.attachment_ratio)
                # 3.3.3 backward
                # divide by grad_accumulation_steps
                # in place
                
                # FIXME: if we want to the sum of a whole batch (batch_size * grad_accumulation_steps), this division is not needed
                # loss.div_(train_args.gradient_accumulation_steps)
                
                # def print_grad_memory(model):
                #     total_bytes = 0
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             param_bytes = param.grad.detach().numel() * param.grad.detach().element_size()
                #             total_bytes += param_bytes
                #             # print(f"{name}: {param.grad.numel()} elements, {param_bytes / (1024**2):.2f} MB")
                #     print(f"Total gradient memory: {total_bytes / (1024**2):.2f} MB")
                # print_grad_memory(model)
                # print(f"Allocated GPU MEMORY: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                # print(f"Reserved GPU MEMORY: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
                # torch.cuda.empty_cache()
                # print(loss.sum())
                loss.backward() # NOTE: WE DON'T NEED TO DIVIDE BY SUM OF LENGTHS IN BACKWARD otherwise the scale <<1
                
                
                if (general_step + 1) % train_args.gradient_accumulation_steps == 0:
                    # 3.3.4 clip grad norm
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm).sum()
                    
                    # if num_params_grad == 0:
                    #     logging.warning("No parameters with gradients found, skipping gradient clipping.")
                    # else:
                    #     logging.debug(f"Gradient norm: {norm.item()}, Average gradient norm: {norm.item() / num_params_grad}")
                    if torch.isnan(norm).any():
                        raise ValueError("Gradient norm is NaN")
                    if torch.isinf(norm).any():
                        raise ValueError("Gradient norm is Inf")
                    # if norm.item() / n_params > train_args.max_grad_norm:
                    #     logging.warning(f"Gradient norm {norm.item() / n_params} is larger than max_grad_norm {train_args.max_grad_norm}")
                    wandb.log({
                        "grad_norm": norm.item(),
                        "average_grad_norm": norm.item() / num_params_grad if num_params_grad > 0 else 0, # for logging purposes
                    }, step=general_step + 1) # log gradient norm
                    # 3.3.5 step
                    # torch.cuda.empty_cache()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # 3.3.6 detach forward inputs and move back to cpu
                data = data.detach().cpu()
                target = target.detach().cpu()
                stack_tape = stack_tape.detach().cpu()
                attachment_labels = attachment_labels.detach().cpu()
                
                # 3.3.7 eval & log
                # print
                # torch.set_grad_enabled(False)
                # loss_w = loss_w.detach().cpu()
                # loss_a = loss_a.detach().cpu()
                # loss = loss.detach().cpu()
                
                # no need grad from now on
                with torch.no_grad():
                    sum_of_seq_lengths_item = torch.sum(batch["lengths"]).item()
                    # assert sum_of_seq_lengths_item == torch.sum(target != model_args.pad_id).item(), f"sum_of_seq_lengths_item: {sum_of_seq_lengths_item}, target: {target}"
                    if (general_step + 1) % train_args.log_interval == 0:

                        logging.info("=" * 100)
                        logging.info(f"Epoch: {epoch}, Step: {step}, Global Step (+1): {general_step + 1},")
                        # Loss Sum: {loss.item()}, Loss Word: {loss_w.item()}, Loss Attachment: {loss_a.item()},
                        logging.info(f"Loss Word for PPL: {loss_w.item() / sum_of_seq_lengths_item}, Loss Attachment for PPL: {loss_a.item() / sum_of_seq_lengths_item}")
                        logging.info(f"Loss weighted: {loss.item() / sum_of_seq_lengths_item}, LR: {scheduler.get_last_lr()[0]}")
                        logging.info(f"PPL this time: {torch.exp(torch.tensor((loss_w + loss_a).item() / sum_of_seq_lengths_item))}")

                    wandb.log({
                        "train_loss": loss.item() / sum_of_seq_lengths_item,
                        "train_loss_word": loss_w.item() / sum_of_seq_lengths_item,
                        "train_loss_attachment": loss_a.item() / sum_of_seq_lengths_item,
                        "train_loss_joint": (loss_w + loss_a).item() / sum_of_seq_lengths_item,
                        "train_joint_ppl": torch.exp(torch.tensor((loss_w + loss_a).item() / sum_of_seq_lengths_item)),
                        "epoch": epoch,
                        "step": step,
                        "global_step": general_step + 1,
                        "lr": scheduler.get_last_lr()[0],
                    }, step=general_step + 1)
                    del loss, loss_w, loss_a, sum_of_seq_lengths_item
                    del target, data, stack_tape, attachment_labels, ids, batch
                    
                    # eval w/ dev set
                    # if best then save
                    if (general_step + 1) % train_args.eval_interval == 0:
                        logging.info("=" * 100)
                        logging.info(f"Eval at Epoch: {epoch}, Step: {step}, Global step (+1): {general_step + 1}")
                        # with torch.no_grad():
                        model.eval()
                        eval_losses_w = []
                        eval_losses_a = []
                        n_words = 0
                        for _, eval_batch in enumerate(dev_dataloader):
                            ids = eval_batch["ids"]
                            ids = torch.cat([torch.full((ids.shape[0], 1), model_args.bos_id), ids], dim=1)
                            target = ids[:, 1:].to(device)
                            data = ids[:, :-1].to(device)
                            stack_tape = eval_batch["stack_tape"].to(device)
                            attachment_labels = eval_batch["attachment_labels"].to(device)
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
                            eval_losses_w.append(loss_w.item())
                            eval_losses_a.append(loss_a.item())
                            n_words += torch.sum(eval_batch["lengths"]).item()
                        eval_loss_w_sum = np.sum(eval_losses_w)
                        eval_loss_a_sum = np.sum(eval_losses_a)
                        # sum_of_all_seq_lengths = np.sum([torch.sum(eval_batch["lengths"]).item() for eval_batch in dev_dataloader])
                        # assert sum_of_all_seq_lengths == n_words, f"sum_of_all_seq_lengths: {sum_of_all_seq_lengths}, n_words: {n_words}"
                        eval_loss_w = eval_loss_w_sum / n_words
                        eval_loss_a = eval_loss_a_sum / n_words
                        eval_loss = eval_loss_w + eval_loss_a
                        word_ppl = torch.exp(torch.tensor(eval_loss_w))
                        all_ppl = torch.exp(torch.tensor(eval_loss))
                        logging.info(f"Eval Loss: {eval_loss}, Word PPL: {word_ppl.item()}, Joint PPL: {all_ppl.item()}")
                        # save best model
                        if eval_loss < best_eval_loss:
                            logging.info(f"Old best eval loss: {best_eval_loss}, new best eval loss: {eval_loss}")
                            best_eval_loss = eval_loss
                            os.makedirs(f"./ckpt/{train_args.run_name}/best", exist_ok=True)
                            # del old model.pt
                            if os.path.exists(f"./ckpt/{train_args.run_name}/best/model.pt"):
                                os.remove(f"./ckpt/{train_args.run_name}/best/model.pt")
                            if isinstance(model, torch.nn.DataParallel):
                                torch.save(model.module.state_dict(), f"./ckpt/{train_args.run_name}/best/model.pt")
                            else:
                                torch.save(model.state_dict(), f"./ckpt/{train_args.run_name}/best/model.pt")
                            # write a file about best eval loss and step (json)
                            with open(f"./ckpt/{train_args.run_name}/best/best_eval.json", "w") as f:
                                ddict = {
                                    "best_eval_joint_loss": best_eval_loss,
                                    "best_eval_word_loss": eval_loss_w,
                                    "best_eval_attachment_loss": eval_loss_a,
                                    "best_eval_joint_ppl": all_ppl.item(),
                                    "best_eval_word_ppl": word_ppl.item(),
                                    "epoch": epoch,
                                    "step": step,
                                    "global_step": general_step + 1,
                                }
                                json.dump(ddict, f)
                            logging.info(f"Best model saved at Epoch: {epoch}, Step: {step}, Global step (+1): {general_step + 1}")
                        model.train()
                        wandb.log({
                            "eval_loss_joint": eval_loss,
                            "eval_loss_word": eval_loss_w,
                            "eval_loss_attachment": eval_loss_a,
                            "eval_joint_ppl": all_ppl.item(),
                            "eval_word_ppl": word_ppl.item(),
                            "best_eval_loss_joint": best_eval_loss,
                            "best_joint_ppl": torch.exp(torch.tensor(best_eval_loss)).item(),
                        }, step=general_step + 1) # log eval loss and ppl
                    # logging.info("=" * 100)
                    # torch.cuda.empty_cache()
            # torch.set_grad_enabled(True)    
            # gc.collect()
        
        # save model
        logging.info("=" * 100)
        logging.info(f"Saving model at Epoch: {epoch}")
        os.makedirs(f"./ckpt/{train_args.run_name}/epoch_{epoch}", exist_ok=True)
        torch.save(model.state_dict(), f"./ckpt/{train_args.run_name}/epoch_{epoch}/model.pt")
        # logging.info("=" * 100)

    logging.info("=" * 100)

    
    del model
    wandb.finish()

    
def main_ddp(model_args: ModelConfig, train_args: TrainConfig, parallel_args: ParallelConfig, rank: int, world_size: int):
    raise NotImplementedError("Distributed Data Parallel (DDP) is not implemented yet.")
    # 0. load datasets
    train_dataset = HFDataset.load_from_disk(
        "./data/BLLIP_LG_train"
    )
    dev_dataset = HFDataset.load_from_disk(
        "./data/BLLIP_LG_dev"
    )
    
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    # wait


if __name__ == '__main__':
    if args.debug:
        # debug mode
        from config import DebugTrainConfig, DebugParallelConfig
        model_args, train_args, parallel_args = ModelConfig(), DebugTrainConfig(), DebugParallelConfig() # adjust by editing py
    else:
        model_args, train_args, parallel_args = ModelConfig(), TrainConfig(), ParallelConfig() # adjust by editing py
    # print args
    from dataclasses import asdict
    from pprint import pprint
    print("=" * 100)
    print("Model Args:")
    pprint(asdict(model_args))
    print("-" * 100)
    print("Train Args:")
    pprint(asdict(train_args))
    print("-" * 100)
    print("Parallel Args:")
    pprint(asdict(parallel_args))
    # print("=" * 100)
    # exit()
    
    # do wandb
    wandb.init(
        project="push_bllip_con",
        name=train_args.run_name,
        config={
            "model_args": asdict(model_args),
            "train_args": asdict(train_args),
            "parallel_args": asdict(parallel_args),
        },
    )
    wandb.run.name = train_args.run_name
    wandb.run.save() # save model_args, train_args, parallel_args

    # main
    # log things in main
    main(model_args, train_args, parallel_args)