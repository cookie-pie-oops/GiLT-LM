# ---------------------------------------------------------
# main_marginal_eval.py
# ---------------------------------------------------------
import torch, json, gc, logging, numpy as np
from datasets import Dataset as HFDataset
from beam_search_utils import BeamSearchDepthBased
from model_bllip_con import PushdownTransformerConstituency
from config import ModelConfig, TrainConfig, ParallelConfig
from collate import collate_fn
import sentencepiece as spm
from tqdm import tqdm
import random, os
DEBUG = False
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="push_bllip_con_test_gas1")
parser.add_argument("--max_stack_depth", type=int, default=150)
script_args = parser.parse_args()
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
    # beam search only accepts batch_size=1
    dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
    )
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # shut down the grad
    torch.set_grad_enabled(False)
    # ---------- 2. model ----------
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
        max_stack_depth=script_args.max_stack_depth,
    ).to(device)

    if parallel_args.parallel == "dp" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ckpt = f"./ckpt/{script_args.run_name}/best/model.pt"
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(model, torch.nn.DataParallel):
        # if module. not in state_dict.keys():
        first_key = next(iter(state_dict.keys()))
        if "module." not in first_key:
            # single gpu
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
    
    # ---------- 3. Beam Search ----------
    with torch.no_grad():
        total_log_p_hat = 0.0     # accumulate log sum_beam p(x,y)
        total_tokens   = 0
        pbar = tqdm(total=len(dl), desc="Testing", disable=DEBUG)
        for one_sample_batch in dl:
            # print(1)
            beam_searcher = BeamSearchDepthBased(beam_size=beam_size)
            
            
            # ids: [1, seq_len]
            ids = one_sample_batch["ids"].to(device)
            # prep: add bos_id
            ids = torch.cat([
                torch.full((1,1), model_args.bos_id, device=device),
                ids,
            ], dim=1) # [1, seq_len+1] WITH BOS AND EOS
            
            _, log_p_hat = beam_searcher(model, ids)   # have returned log sum_beam p(x,y)

            # candidate_scores = [b.score for b in beams]
            # joint_ppls = [torch.exp(torch.tensor(-b.score / (ids.shape[1]-1))).item() for b in beams]
            # one_ppl = torch.exp(torch.tensor(-log_p_hat / (ids.shape[1]-1))).item()
            one_ppl = np.exp(-log_p_hat.item() / (ids.shape[1]-1))

            total_log_p_hat += log_p_hat.item()
            total_tokens    += ids.shape[1] - 1
            running_ppl = np.exp(-total_log_p_hat / total_tokens)
            
            pbar.set_postfix(
                log_p_hat=log_p_hat.item(),
                total_log_p_hat=total_log_p_hat,
                total_tokens=total_tokens,
                one_ppl=one_ppl,
                running_ppl=running_ppl,
                # gold_log_p_hat=gold_log_p_hat,
                # gold_one_ppl=gold_one_ppl,
                length=ids.shape[1]-1,
            )

            pbar.update(1)
            del ids, log_p_hat, running_ppl, beam_searcher
        pbar.close()
        

    # ---------- 4. PPL ----------
    total_nll = - total_log_p_hat
    ppl = np.exp(total_nll / total_tokens)

    logging.info(f"[Marginal] Log Likelihood Sum = {total_log_p_hat:.4f} , "
                 f"# of tokens = {total_tokens}, Marginal PPL = {ppl:.4f}")

    # ---------- 5. saving ----------
    out_path = f"./ckpt/{script_args.run_name}/best/test_beam{beam_size}.json"
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

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    #     handlers=[
    #         logging.FileHandler(f"./ckpt/{script_args.run_name}/best/test_beam.log"),
    #         logging.StreamHandler()
    #     ]
    # )

    eval_marginal(model_args, train_args, parallel_args, beam_size=300)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   