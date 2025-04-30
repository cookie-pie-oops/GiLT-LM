
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
    
if __name__ == "__main__":
    # seed_everything(42)

    configure_logger('logs/beam_txl.log')
    logger = get_logger()
    vocab, vocab_size, word2idx, bos, eos, pad_id, left_arc, right_arc, pop_root, startofword_id = load_vocab('tokenizer/spm.vocab')
    # 0,1,2 actions. 3~9 terminals
    BATCH = 1
    BEAM = 100
    WORD_BEAM = 10
    SHIFT = 0
    VOCAB = vocab_size
    MAX_LEN = 500
    torch.manual_seed(123456)
    np.random.seed(123456)
    ranges = TokenTypeRanges(bos, pad_id, vocab_size, left_arc, right_arc)
    vocab_meta = VocabMeta([left_arc, right_arc + 1])
    # subtoken_begin = torch.randn(BATCH, MAX_LEN // 3) < 0.5  # [:, 0]  is always a begin
    original = []
    # original_startofword = []
    subtoken_begins = []
    original_length = []
    original_seq_length = []
    checkpoint = torch.load('models/standard_rbt_txl_1.pt')
    model = checkpoint['model']
    model.eval()
    model.cuda()
    sp = spm.SentencePieceProcessor(model_file='tokenizer/spm.model')
    file_list = os.listdir("test_suites/json/.")
    # print(file_list)
    for file in file_list:
        test_suite_parser = TestSuiteParser(file[:-5])
        logger.info(file[:-5])
        BEAM = 100
        WORD_BEAM = 10
            
        # print(test_suite_parser.meta_data["formula"])
        for idx in tqdm(range(len(test_suite_parser.meta_data["data"]))):
            examples = test_suite_parser.get_example(idx)
            phen2surprisals = {}
            for phen in examples:
                # logger.info(examples[phen])
                encoded = sp.Encode(examples[phen] + ["."], out_type=int)
                # logger.info(sp.Encode(" ".join(examples[phen]), out_type=int))
                tgt_idx = []
                encoded.insert(0, [bos])
                encoded.append([pop_root])
                encoded.append([eos])
                # logger.info(encoded)
                word_idx = -1
                prev_idx = -1
                for word in encoded:
                    word_idx += len(word)
                    tgt_idx.append((prev_idx, word_idx))
                    prev_idx = word_idx 
                tgt_idx = tgt_idx[1:-1]
                encoded = [x for word in encoded for x in word]

                # target_surprisals, logprobs, target_idxs, _ = evaluator.get_surprisals(
                #     examples[phen]
                # )
                subtoken_begin = [1 if startofword_id[word] == 1 else 0 for word in encoded]
                subtoken_begin[-2] = 1 # for pop_root
                subtoken_begin[-1] = 1 # for eos
                tokens = torch.LongTensor(encoded).cuda().reshape(1, -1)
                word_num = sum(subtoken_begin) - 2 # -2 for pop_root and eos 
                subtoken_begin = torch.BoolTensor(subtoken_begin).cuda().reshape(1, -1)
                # tokens = tokens[:, :5]
                # subtoken_begin = subtoken_begin[:, :5]
                # print(subtoken_begin)
                # exit()
                MAX_LEN = len(tokens[0]) + (word_num - 1)
                beams, beam_token_scores, beam_scores, beam_sum_scores = word_sync_beam_search(
                    model, ranges, tokens, subtoken_begin, startofword_id, vocab_size, BEAM, WORD_BEAM, SHIFT, MAX_LEN, 15, vocab_meta
                )
                # logger.info(beams[:, 0:10, :])
                # logger.info(beam_scores[:, 0:10])
                # logger.info(beam_token_scores[:, 0:10, :].sum(-1))
                # exit()
                scores = -beam_sum_scores.cpu().numpy() # - means surprisals
                target_surprisals = [scores[0][tgt_idx[i][1]] - scores[0][tgt_idx[i][0]] for i in range(len(tgt_idx))]
                # print(target_surprisals)
                # logger.info(target_surprisals)
                phen2surprisals[phen] = [0] + target_surprisals

            extracted_formula = test_suite_parser.extract_formulas(phen2surprisals)
            test_suite_parser.answers[idx] = extracted_formula
        
        acc = 0.0
        for formula in test_suite_parser.answers:
            answer = eval_math_expr(formula)
            logger.info(f"score: {answer}")
            acc += answer
        
        logger.info(f"correct rate: {acc / len(test_suite_parser.answers)}")