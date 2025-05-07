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
verbose = True

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
def eval_sg(model_args: ModelConfig,
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
        max_stack_depth=model_args.max_stack_depth
    ).to(device)
    
    if parallel_args.parallel == "dp" and torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        logging.warning("This script does not support DataParallel.")
    # train_args.run_name = "push_bllip_con_test_gas4_only_1_pushdown"
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

    file_list = os.listdir("test_suites/json/.")
    final_acc = []
    final_acc_per_type = {}
    logging.info("# of tests: {}".format(len(file_list)))
    for ii, file in enumerate(file_list):
        test_suite_parser = TestSuiteParser(file[:-5])
        logging.info("-" * 100)
        logging.info("Testing on file: {}".format(file[:-5]))
        logging.info("Test number: {}".format(ii))
        suite_type = get_suite_type(file[:-5])
        acc = 0.0
        logging.info(f"Suite type: {suite_type}")
        logging.info(f"Formula: {test_suite_parser.meta_data['formula']}")
       
        # ---------- 3. Beam Search ----------
        for sample_idx in tqdm(range(len(test_suite_parser.meta_data["data"])), desc=f"Evaluating {file}", disable=verbose):
            examples = test_suite_parser.get_example(sample_idx)
            phen2surprisals = {}
            beam_searcher = BeamSearchDepthBased(beam_size=beam_size, gold_attach=None)
            for phen in examples:
                sent_list = examples[phen]
                # if sent_list[-1] and sent_list[-1][-1] not in [".", "!", "?"]:
                #     sent_list += ["."]
                ids_ori = sp.Encode(sent_list, out_type=int) # `encoded` in reference script
                
                
                ids_ori.insert(0, [model_args.bos_id])
                ids_ori.append([model_args.eos_id])
                # ids_ori: [[1], [76], [7268], [74], [63], [5865], [60, 8063], [7400, 73], [60, 61], [2]]
                # Compute the start and end indices of each word in the flattened token sequence
                tgt_idx = []
                word_idx = -1
                prev_idx = -1
                for word in ids_ori:
                    word_idx += len(word)
                    tgt_idx.append((prev_idx, word_idx))
                    prev_idx = word_idx
                tgt_idx = tgt_idx[1:] # remove the first and last one, which are bos and eos
                # breakpoint()
                # e.g.
                # tgt_idx: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 9), (9, 11)]
                # ids_ori (not processed, w/o bos and eos): [[76], [7268], [74], [63], [5865], [60, 8063], [7400, 73], [60, 61]]
                # one-to-one mapping, each token has a list in ids, 
                # which is mapped to a span [inclusive, exclusive] in tgt_idx
                ids = [x for word in ids_ori for x in word] # flatten
                # ids: [1, 76, 7268, 74, 63, 5865, 60, 8063, 7400, 73, 60, 61, 2]
                is_start_of_word = [1 if start_of_word_bool[word] == 1 else 0 for word in ids]
                is_start_of_word[-1] = 1 # eos
                ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device) # shape [1, L+1] where L=seqlen is the length of the sentence with one of bos/eos
                # seqlen = ids.shape[1] - 1
                is_start_of_word = torch.BoolTensor(is_start_of_word).unsqueeze(0).to(device) # shape [1, L+1]
                # search for the whole seq, beam_size
                
                
                # --- NOTE: Things below can be freely adapted to other beam search implementations ---
                _, _, prefix_trajectories = beam_searcher(model, ids, return_trail=True) # where -1*score = surprisal
                prefix_trajectories = (-torch.tensor(prefix_trajectories)).tolist() # -> prefix list of surprisals
                prefix_trajectories = [0] + prefix_trajectories # add the first one
                # prefix_trajectories: [0=-log p(x0), -log p(x1|x0), -log p(x2|x0,x1), ...]
                # breakpoint()
                target_surprisals = [
                    prefix_trajectories[tgt_idx[i][1]] - prefix_trajectories[tgt_idx[i][0]]
                    for i in range(len(tgt_idx))
                ] # -log p(xt | x<xt)
                # breakpoint()
                # --- Things above can be freely adapted to other beam search implementations ---
                phen2surprisals[phen] = [0] + target_surprisals
                
            
            extracted_formula = test_suite_parser.extract_formulas(phen2surprisals)
            test_suite_parser.answers[sample_idx] = extracted_formula
            answer = eval_math_expr(extracted_formula)
            if not math.isnan(answer):
                acc += answer
                if verbose:
                    logging.info(f"Example {sample_idx}/{len(test_suite_parser.meta_data["data"])}: {examples} -> [{extracted_formula}] == {answer}")
                    logging.info(f"True/All: {acc}/{sample_idx + 1}")
            else:
                logging.error(f"Invalid formula: {extracted_formula}")
                raise ValueError(f"Invalid formula: {extracted_formula}")
        
        acc /= len(test_suite_parser.answers) if len(test_suite_parser.answers) > 0 else 0.
        final_acc.append(acc)
        if suite_type not in final_acc_per_type:
            final_acc_per_type[suite_type] = [acc]
        else:
            final_acc_per_type[suite_type].append(acc)
        logging.info(f"Accuracy for {file[:-5]}: {acc}")
    logging.info("=" * 100)
    logging.info(f"[Syntactic Generalization] Final accuracy: {sum(final_acc) / len(final_acc)}")
    for suite_type, acc_list in final_acc_per_type.items():
        logging.info(f"Accuracy for {suite_type}: {sum(acc_list) / len(acc_list)}")
    logging.info("=" * 100)
    
    # also add the sg to json
    final_acc_per_type["overall"] = sum(final_acc) / len(final_acc)
    final_acc_per_type["unreduced"] = final_acc
    out_path = f"./ckpt/{train_args.run_name}/best/test_sg.json"
    with open(out_path, "w") as f:
        json.dump(final_acc_per_type, f, indent=4)

if __name__ == "__main__":
    model_args, train_args, parallel_args = ModelConfig(), TrainConfig(), ParallelConfig()
    eval_sg(model_args, train_args, parallel_args, beam_size=300)
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Done.")   