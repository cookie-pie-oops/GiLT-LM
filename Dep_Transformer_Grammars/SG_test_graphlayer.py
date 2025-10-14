from typing import List
import torch
from dataclasses import dataclass
# from lightning.pytorch import seed_everything
import numpy as np
import numba
import numba.cuda as cuda
import argparse
import time
from helping_utils.logger import configure_logger, get_logger
from model_bllip_dep import TransformerGrammar, BiaffineAttention, calculate_depth, dijkstra
from beam_search_utils import update_beam, load_vocab, load_data, add_to_all
from copy import deepcopy
import json
import re
import os
from tqdm import tqdm
import sentencepiece as spm
import math
import heapq
from itertools import count
import itertools

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

    def evaluate_example(self, idx, evaluator, verbose=False):
        examples = self.get_example(idx)
        phen2surprisals = {}
        for phen in examples:
            
            target_surprisals, logprobs, target_idxs, _ = evaluator.get_surprisals(
                examples[phen]
            )
            if verbose:
                print("Regions: {}".format(examples[phen]))
                print(logprobs)
            phen2surprisals[phen] = [0] + target_surprisals

        extracted_formula = self.extract_formulas(phen2surprisals)
        self.answers[idx] = extracted_formula

    def evaluate_all(self, evaluator=None):
        for idx in tqdm(range(len(self.meta_data["data"]))):
            self.evaluate_example(idx, evaluator)
        return

def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models/graphlayer.pt', type=str)
parser.add_argument('--beamsize', default=100, type=int)
parser.add_argument('--scorebeamsize', default=20, type=int)

if __name__ == "__main__":
    # seed_everything(42)
    args = parser.parse_args()
    counter = count()
    configure_logger('logs/sg_test.log')
    logger = get_logger()
    vocab_size, pad, bos, eos, startofword_id, vocab = load_vocab('../data_process/spm_parsing/BLLIP_spm.vocab')
    torch.manual_seed(123456)
    np.random.seed(123456)
    original = []
    subtoken_begins = []
    original_length = []
    original_seq_length = []

    model_path = args.model_path
    beamsize = args.beamsize
    scorebeamsize = args.scorebeamsize
    logger.info("Model path: {}".format(model_path))
    logger.info("Beam size: {}".format(beamsize))
    logger.info("Score beam size: {}".format(scorebeamsize))
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model = checkpoint['model']
    biaffine_model = checkpoint['biaffine_model']
    model.eval()
    biaffine_model.eval()
    model.to(device)
    biaffine_model.to(device)

    biaffine_model.set_temperature(1.0)
    
    sp = spm.SentencePieceProcessor(model_file='../data_process/spm_parsing/BLLIP_spm.model')
    file_list = os.listdir("test_suites/json/.")
    final_acc = []
    type_acc = {}
    for file in file_list:
        test_suite_parser = TestSuiteParser(file[:-5])
        logger.info(file[:-5])
        if "npi" in file or "reflexive" in file:
            suite_type = "licensing"
        elif "mvrr" in file or "npz" in file:
            suite_type = "garden-path effects"
        elif "center_embed" in file:
            suite_type = "center embedding"
        elif "subordination" in file:
            suite_type = "gross syntactic expectation"
        elif "fgd" in file or "cleft" in file:
            suite_type = "long-distance dependencies"
        elif "number" in file:
            suite_type = "agreement"
        else:
            suite_type = "Other"
        for idx in range(len(test_suite_parser.meta_data["data"])):
            examples = test_suite_parser.get_example(idx)
            phen2surprisals = {}
            for phen in examples:
                encoded = sp.Encode(examples[phen] + ["."], out_type=int)
                tgt_idx = []
                encoded.insert(0, [1])
                encoded.append([2])
                word_idx = -1
                prev_idx = -1
                for word in encoded:
                    word_idx += len(word)
                    tgt_idx.append((prev_idx, word_idx))
                    prev_idx = word_idx 
                tgt_idx = tgt_idx[1:-1]
                encoded = [x for word in encoded for x in word]
                start_predict_new_word = [1 if startofword_id[encoded[i + 1]] else 0 for i in range(len(encoded) - 1)]
                sent_index_to_id = []
                count_num = 0
                for word_id in encoded:
                    if word_id in [bos, eos]:
                        sent_index_to_id.append(-1)
                        continue
                    elif startofword_id[word_id] == 1:
                        count_num += 1
                    sent_index_to_id.append(count_num)
                
                scores, _ = update_beam(encoded, model, biaffine_model, start_predict_new_word,
                    sent_index_to_id, beamsize, scorebeamsize, device, logger)

                # _, prob = model([encoded], None, None, use_mask=None)
                # prob = prob.log_softmax(-2)
                # scores = [-prob[i, encoded[i + 1]].item() for i in range(len(encoded[1:]))]
                # scores.insert(0, 0)
                # scores = list(itertools.accumulate(scores))        

                # import pdb;pdb.set_trace()
                target_surprisals = [scores[tgt_idx[i][1]] - scores[tgt_idx[i][0]] for i in range(len(tgt_idx))]
                # print(target_surprisals)
                # logger.info(target_surprisals)
                phen2surprisals[phen] = [0] + target_surprisals
            
            extracted_formula = test_suite_parser.extract_formulas(phen2surprisals)
            test_suite_parser.answers[idx] = extracted_formula
        acc = 0.0
        res_list = []
        for formula in test_suite_parser.answers:
            answer = eval_math_expr(formula)
            res_list.append(str(answer))
            acc += answer
        logger.info(f"{('|').join(res_list)}")

        logger.info(f"correct rate: {acc / len(test_suite_parser.answers)}")
        final_acc.append(acc / len(test_suite_parser.answers))
        logger.info(f"mean correct rate up to now: {np.mean(final_acc)}")
        type_acc.setdefault(suite_type, []).append(acc / len(test_suite_parser.answers))
    
    logger.info(f"final correct rate: {np.mean(final_acc)}\n")
    SG_Score = 0.0
    for key, value in type_acc.items():
        logger.info(f"{key}: {np.mean(value)}")
        if key != "Other":
            SG_Score += np.mean(value)
    logger.info(f"SG Score: {SG_Score / (len(type_acc) - 1)}")