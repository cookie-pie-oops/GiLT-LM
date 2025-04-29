# ---------------------------------------------------------
# main_marginal_eval.py
# ---------------------------------------------------------
import torch, json, gc, logging, numpy as np
from datasets import Dataset as HFDataset
from beam_search_utils import BeamSearchDepthBased           # 记得换成你真实文件名
from model_bllip_con import PushdownTransformerConstituency
from config import ModelConfig, TrainConfig, ParallelConfig
from collate import collate_fn
import sentencepiece as spm
from tqdm import tqdm
DEBUG = True

@torch.inference_mode()
def eval_marginal(model_args: ModelConfig,
                  train_args: TrainConfig,
                  parallel_args: ParallelConfig,
                  beam_size: int = 300):

    # ---------- 1. 数据 ----------
    test_ds = HFDataset.load_from_disk("./data/BLLIP_LG_test")
    # max length in test set using test_ds["lengths"]
    # max_len = max(test_ds["lengths"])
    # print(max_len)
    # if debug then 10 samples
    if DEBUG:
        test_ds = test_ds.select(range(10))
    # beam search 只能一条句子一条句子地跑，batch_size=1
    dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )

    # ---------- 2. 模型 ----------
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

    if parallel_args.parallel == "dp" and torch.cuda.device_count() > 1:
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

    # ---------- 3. Beam Search ----------
    with torch.no_grad():
        beam_searcher = BeamSearchDepthBased(beam_size=beam_size)

        total_log_p_hat = 0.0     # 累加 log Σ_beam p(x,y)
        total_tokens   = 0
        pbar = tqdm(total=len(dl), desc="Testing")
        for one_sample_batch in dl:
            # ids: [1, seq_len]
            ids = one_sample_batch["ids"].to(device)
            # prep: add bos_id
            ids = torch.cat([
                torch.full((1,1), model_args.bos_id, device=device),
                ids,
            ], dim=1) # [1, seq_len+1]

            _, log_p_hat = beam_searcher(model, ids)   # 已经返回 log Σ_beam p(x,y)

            total_log_p_hat += log_p_hat.item()
            total_tokens    += ids.numel() - 1         # 不算 bos
            running_ppl = np.exp(-total_log_p_hat / total_tokens)
            torch.cuda.empty_cache()
            pbar.set_postfix(
                log_p_hat=log_p_hat.item(),
                total_log_p_hat=total_log_p_hat,
                total_tokens=total_tokens,
                running_ppl=running_ppl,
            )
            del ids, log_p_hat, running_ppl
            pbar.update(1)
        pbar.close()
        

    # ---------- 4. PPL ----------
    total_nll = - total_log_p_hat
    ppl = np.exp(total_nll / total_tokens)

    logging.info(f"[Marginal] log_sum_p = {total_log_p_hat:.4f} , "
                 f"tokens = {total_tokens} , PPL = {ppl:.4f}")

    # ---------- 5. 保存 ----------
    out_path = f"./ckpt/{train_args.run_name}/best/test_beam{beam_size}.json"
    with open(out_path, "w") as f:
        json.dump({
            "beam_size": beam_size,
            "log_sum_p": total_log_p_hat,
            "tokens": total_tokens,
            "ppl": ppl
        }, f, indent=2)

# ------------------------------------------------------------------
if __name__ == "__main__":
    model_args    = ModelConfig()
    train_args    = TrainConfig()
    parallel_args = ParallelConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"./ckpt/{train_args.run_name}/best/test_beam.log"),
            logging.StreamHandler()
        ]
    )

    eval_marginal(model_args, train_args, parallel_args, beam_size=300)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   