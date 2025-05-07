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
import random, os
DEBUG = False

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
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # shut down the grad
    torch.set_grad_enabled(False)
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

    ckpt = f"./ckpt/{train_args.run_name}/epoch_3/model.pt"
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
    
    # ---------- 3. Beam Search ----------
    with torch.no_grad():
        total_log_p_hat = 0.0     # 累加 log Σ_beam p(x,y)
        total_tokens   = 0
        pbar = tqdm(total=len(dl), desc="Testing", disable=DEBUG)
        for one_sample_batch in dl:
            # gold_attach = one_sample_batch["attachment_labels"] # [1, seq_len]
            # print(1)
            beam_searcher = BeamSearchDepthBased(beam_size=beam_size, gold_attach=None)
            
            
            # ids: [1, seq_len]
            ids = one_sample_batch["ids"].to(device)
            # prep: add bos_id
            ids = torch.cat([
                torch.full((1,1), model_args.bos_id, device=device),
                ids,
            ], dim=1) # [1, seq_len+1] WITH BOS AND EOS
            
            beams, log_p_hat = beam_searcher(model, ids)   # 已经返回 log Σ_beam p(x,y)

            # candidate_scores = [b.score for b in beams]
            # joint_ppls = [torch.exp(torch.tensor(-b.score / (ids.shape[1]-1))).item() for b in beams]
            # one_ppl = torch.exp(torch.tensor(-log_p_hat / (ids.shape[1]-1))).item()
            one_ppl = np.exp(-log_p_hat.item() / (ids.shape[1]-1))

            total_log_p_hat += log_p_hat.item()
            total_tokens    += ids.shape[1] - 1
            running_ppl = np.exp(-total_log_p_hat / total_tokens)
            
            if DEBUG:
                time_left = pbar.format_dict["remaining"]
                logging.info(f"Log P: {log_p_hat.item():.4f} , "
                            f"One PPL: {one_ppl:.4f} , "
                            f"Running PPL: {running_ppl:.4f} , "
                            f"Total Log P: {total_log_p_hat:.4f} , "
                            f"Total Tokens: {total_tokens}, "
                            f"Length: {ids.shape[1]-1}, "
                            f"Time Left: {time_left}")
                beam_scores = [b.score for b in beams]
                logging.info(f"Sentence: {sp.decode(ids[0].cpu().tolist())}")
                logging.info(f"Pieces: {sp.id_to_piece(ids[0].cpu().tolist())}")
                logging.info(f"Beam Scores: {beam_scores}")
                # exit()
            
            # torch.cuda.empty_cache()


            # start to eval gold
            # ---- 1.2 初始化 stack / depth / reduced 状态 ----
            # seqlen = ids.shape[1]
            # stack_tape   = torch.zeros((1, seqlen, seqlen), device=device, dtype=torch.long)
            # list_reduced = [set()]
            # stacks_hist  = [[[0]]]              # 开始时栈里只有 <bos>

            # gold_log_p_hat = 0.0
            # stack_one_row = torch.zeros((1, seqlen), device=device, dtype=torch.long)
            # # ---- 1.3 逐步累加 log 概率 ----
            # for step in range(1, seqlen):      # step=1 ... L
            #     lp_w, lp_attach = model.take_step_silver_tree(
            #         ids,                                    # [1, L+1]
            #         stack_tape[:, :step, :step],            # [1, step, step]
            #         list_reduced,
            #         step
            #     )                                          # 返回 [1] , [1, step+1]

            #     gold_a = gold_attach[0][step-1]                     # gold attach 决策
            #     # breakpoint()
            #     gold_log_p_hat += lp_w.item() + lp_attach[0, gold_a].item()

            #     # ---- 1.4 用 gold 决策推进栈、depth、reduced ----
            #     # 下面这行用的是你在 beam 里已有的 _update_stacks()
            #     stacks_hist, list_reduced, stack_one_row = beam_searcher._update_stacks(
            #         reduced_states=list_reduced,
            #         stacks=stacks_hist,
            #         attachment_decisions=[gold_a],
            #         step_prime=step,
            #         depths=stack_one_row,
            #     )
            #     # 把新 depth 行写回总的 stack_tape
            #     stack_tape[:, step, :] = stack_one_row.clone()
            # gold_one_ppl = torch.exp(-torch.tensor(gold_log_p_hat / (seqlen-1))).item()
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
            # del ids, log_p_hat, running_ppl, lp_sent, lp_w, lp_attach, gold_a, beams, candidate_scores, beam_searcher
        pbar.close()
        

    # ---------- 4. PPL ----------
    total_nll = - total_log_p_hat
    ppl = np.exp(total_nll / total_tokens)

    logging.info(f"[Marginal] Log Likelihood Sum = {total_log_p_hat:.4f} , "
                 f"# of tokens = {total_tokens}, Marginal PPL = {ppl:.4f}")

    # ---------- 5. saving ----------
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

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    #     handlers=[
    #         logging.FileHandler(f"./ckpt/{train_args.run_name}/best/test_beam.log"),
    #         logging.StreamHandler()
    #     ]
    # )

    eval_marginal(model_args, train_args, parallel_args, beam_size=300)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   