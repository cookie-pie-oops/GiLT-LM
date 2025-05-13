import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json, os, re
import math
from tqdm import tqdm
import logging
import numpy as np
from config import ModelConfig, TrainConfig, ParallelConfig
from model_bllip_con import PushdownTransformerConstituency
from beam_search_utils import BeamSearchDepthBased
import random
import gc
from utils import get_suite_type, eval_math_expr
verbose = False
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="push_bllip_con_test_gas1")
parser.add_argument("--max_stack_depth", type=int, default=150)
script_args = parser.parse_args()

class TestSuiteParser:
    def __init__(self, test_suite_file):
        self.test_suite_file = test_suite_file
        self.read_test_suite()
        self.answers = [0 for _ in range(len(self.meta_data["data"]))]

    def read_test_suite(self):
        data_file = "test_suites/json/{}.json".format(self.test_suite_file)
        with open(data_file, "r") as f:
            data = json.load(f)
        self.meta_data = {
            "formula": data["predictions"][0]["formula"],
            "data": self.get_sents(data),
        }

    def get_sents(self, data):
        all_ex = []
        for item in data["items"]:
            curr_ex = {}
            for cond in item["conditions"]:
                regions = [x["content"] for x in cond["regions"]]
                curr_ex[cond["condition_name"]] = regions
            all_ex.append(curr_ex)
        return all_ex

    def extract_formulas(self, surprisal_dict):
        formula = self.meta_data["formula"]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            positions = set(re.findall(r"\((\d+);%{}%".format(key), formula))
            for position in positions:
                formula = formula.replace(
                    "({};%{}%)".format(position, key),
                    str(surprisal_dict[key][int(position)]),
                )
        ### replace [ with ( and ] with ) to make it a valid math expression

        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        return formula

    def get_example(self, idx):
        return self.meta_data["data"][idx]

def load_vocab(path):
    vocab_file = path

    pad_id = None
    bos_id = None
    eos_id = None

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
            elif vocab[i].startswith('▁'):
                startofword_id[i] = 1  # NOTE: 1 means we have a new word in this token. this array has a length of vocab_size

    return vocab_size, pad_id, bos_id, eos_id, startofword_id, vocab

@torch.inference_mode()
def eval_blimp(model_args: ModelConfig,
            train_args: TrainConfig,
            parallel_args: ParallelConfig, 
            beam_size: int = 300
            ):
    # ---------- 1. Loading everything ----------
    sp = spm.SentencePieceProcessor()
    sp.Load("./data_process/spm_parsing/BLLIP_spm.model")
    vocab_size = sp.GetPieceSize()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("=" * 100)
    logging.info(f"Testing started")
    
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
        raise NotImplementedError("dp is not implemented yet")
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
    # print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    random.seed(train_args.seed)
    os.environ["PYTHONHASHSEED"] = str(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed_all(train_args.seed)
    # shut down the grad
    torch.set_grad_enabled(False)
    # eval the parallel module
    if parallel_args.parallel == "dp" and torch.cuda.device_count() > 1:
        model.module.eval()
    
    _, pad_id, bos_id, eos_id, start_of_word_bool, vocab = load_vocab('./data_process/spm_parsing/BLLIP_spm.vocab')
    # NOTE: start_of_word_bool is `startofword_id` in the reference script
    assert pad_id == model_args.pad_id
    assert bos_id == model_args.bos_id
    assert eos_id == model_args.eos_id
    # ---------- 2. Loading test data ----------

    file_list = os.listdir("blimp/json/.")
    final_acc = []
    final_acc_per_type = {}
    logging.info("# of tests: {}".format(len(file_list)))
    for ii, file in enumerate(file_list):
        logging.info("-" * 100)
        logging.info("Testing on file: {}".format(file[:-6]))
        logging.info("Test number: {}".format(ii))
        acc = 0.0
        with open("blimp/json/{}".format(file), "r") as f:
            data = f.readlines() # jsonl file
            sample_list = [json.loads(ds.strip()) for ds in data]
        # sample_list = random.sample(sample_list, int(len(sample_list) / 10)) # load 1/10 of the data randomly
        sample_num = (len(sample_list) - 1) // 10 + 1
        sample_index = [x*10 for x in range(sample_num)]
        sample_list = [sample_list[i] for i in sample_index]
        
        # ---------- 3. Beam Search ----------
        pbar = tqdm(total=len(sample_list), desc="file: {}".format(file[:-6]), disable=verbose)
        for sample_idx in range(len(sample_list)):
            examples = {
                "sentence_good": sample_list[sample_idx]["sentence_good"],
                "sentence_bad": sample_list[sample_idx]["sentence_bad"],
            }
            phen2surprisals = {}
            beam_searcher = BeamSearchDepthBased(beam_size=beam_size, gold_attach=None)
            for phen in examples:
                sent_list = examples[phen]
                ids = sp.Encode(sent_list, out_type=int) # flattened
                ids = [bos_id] + ids + [eos_id]
                # to torch tensor
                ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device) # shape [1, N]
                # --- NOTE: Things below can be freely adapted to other beam search implementations ---
                _, logprob = beam_searcher(model, ids, return_trail=False) # where -1*score = surprisal
                # prefix_trajectories: [0=-log p(x0), -log p(x1|x0), -log p(x2|x0,x1), ...]
                # breakpoint()
                # breakpoint()
                # --- Things above can be freely adapted to other beam search implementations ---
                phen2surprisals[phen] = -logprob # surprisal -log p(x0:xN)
                
            
            answer = 1 if phen2surprisals["sentence_good"] < phen2surprisals["sentence_bad"] else 0
            if not math.isnan(answer):
                acc += answer
                if verbose:
                    logging.info(f"Example {sample_idx}/{len(sample_list)}: \"{examples['sentence_good']}\" vs \"{examples['sentence_bad']}\"")
                    logging.info(f"Surprisal: {phen2surprisals['sentence_good']} < {phen2surprisals['sentence_bad']} ? -> {'True' if answer == 1 else 'False'}")
                    logging.info(f"True/All: {acc}/{sample_idx + 1}")
            else:
                logging.error(f"Surprisal is NaN for example {sample_idx}/{len(sample_list)}")
            pbar.set_postfix_str(f"Realtime running accuracy: {acc / (sample_idx + 1):.4f}")
            pbar.update(1)
        pbar.close()
        acc /= len(sample_list)
        final_acc.append(acc)
        logging.info(f"Accuracy for {file[:-6]}: {acc}")
        logging.info(f"Mean running macro accuracy: {sum(final_acc) / len(final_acc)}")
    logging.info("=" * 100)
    logging.info(f"[BLiMP] Final macro accuracy: {sum(final_acc) / len(final_acc)}")
    logging.info("=" * 100)
    
    # also add the sg to json
    final_acc_per_type["overall"] = sum(final_acc) / len(final_acc)
    final_acc_per_type["unreduced"] = final_acc
    out_path = f"./ckpt/{script_args.run_name}/best/test_blimp.json"
    with open(out_path, "w") as f:
        json.dump(final_acc_per_type, f, indent=4)

if __name__ == "__main__":
    model_args, train_args, parallel_args = ModelConfig(), TrainConfig(), ParallelConfig()
    eval_blimp(model_args, train_args, parallel_args, beam_size=300)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   