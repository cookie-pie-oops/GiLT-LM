import os
from tqdm import tqdm
from nltk.tree import Tree
from data_utils import flatten, binarize_tree, post_process_op
import sentencepiece as spm
from attachment_utils import compute_attachment_labels_text
from stack_tape_utils import compute_stack_tape
# HFDataset
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
DEBUG = True

def _list_create(doc_path, dev_early_stop_each=500, test_early_stop_each=1000):  # -1 for no limit
    train_list = []
    dev_list = []
    test_list = []
    years = ['1987', '1988', '1989']

    # if early_stopping < -1:
    #     raise ValueError(
    #         "early_stopping should be -1 or a nonnegative integer")
    dev_dirs = [
        'w7_001', 'w8_001', 'w9_010'
    ]
    test_dirs = [
        'w7_002', 'w8_002', 'w9_011'
    ]
    for year in os.listdir(doc_path):
        dev_cnt = 0
        test_cnt = 0
        if year not in years:
            continue
        year_dir = os.path.join(doc_path, year)
        if not os.path.isdir(year_dir):
            continue
        # traverse the subdirectories
        for sub_dir in tqdm(os.listdir(year_dir), desc=f"Processing year {year}"): # sub_dir be like w7_001
            sub_dir_path = os.path.join(year_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            # print(f"  Processing subdir: {sub_dir}")
            # sort files
            # file_list = os.listdir(sub_dir_path)
            
            file_list = sorted(os.listdir(sub_dir_path))
            for file_name in file_list:
                file_path = os.path.join(sub_dir_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    full_content = "".join(
                        [line.replace("\t", " ").strip("\n") for line in lines])
                    # `(S1` as the separator
                    origin_sentences = full_content.split("(S1")
                    for origin_sentence in origin_sentences:
                        origin_sentence = origin_sentence.strip()
                        if origin_sentence.strip() == "":
                            continue
                        complete_sentence = "(S1" + origin_sentence
                        # dataset_list.append(complete_sentence)
                        if sub_dir in dev_dirs and dev_cnt < dev_early_stop_each:
                            dev_list.append(complete_sentence)
                            dev_cnt += 1
                        elif sub_dir in test_dirs and test_cnt < test_early_stop_each:
                            test_list.append(complete_sentence)
                            test_cnt += 1
                        elif sub_dir not in dev_dirs and sub_dir not in test_dirs:
                            train_list.append(complete_sentence)
                        else:
                            # discard
                            pass
                        # cnt += 1
    return train_list, dev_list, test_list


def parse_and_split(doc_path, sp, dev_num=1500, test_num=3000, early_stopping=-1):

    # doc_path = "../Dataset/bliip_87_89_wsj/"
    # original_list = _list_create(doc_path, early_stopping=early_stopping)
    train_ori, dev_ori, test_ori = _list_create(doc_path, dev_early_stop_each=dev_num//3, test_early_stop_each=test_num//3)
    # split to train,dev,test
    # shuffle
    import random
    # seed
    random.seed(12345)
    random.shuffle(train_ori)
    random.shuffle(dev_ori)
    random.shuffle(test_ori)
    # split
    # train_ori = train_ori
    assert len(dev_ori) == dev_num, "dev set size is not equal to dev_num"
    assert len(test_ori) == test_num, "test set size is not equal to test_num"
    
    def fix_empty_leaf_nodes(t, unk_token="<unk>"):
        if isinstance(t, str):
            # t is a leaf node (string)
            # check if t is empty or contains only whitespace
            if not t.strip():
                return unk_token
            else:
                return t
        else:
            # t is a Tree
            new_children = []
            for child in t:
                fixed_child = fix_empty_leaf_nodes(child, unk_token)
                new_children.append(fixed_child)
            # use the fixed children to replace the original children
            t[:] = new_children
            return t
    
    def parse_sentences(ori_data):
        discard = 0
        in_sentences = []
        parses = []
        for sent in tqdm(ori_data, desc="Parsing sentences"):
            # 如果 sent 不是一个 Tree 对象，尝试将其转换为 Tree
            if not isinstance(sent, Tree):
                try:
                    tree = Tree.fromstring(sent)
                except Exception as e:
                    print("Error: ", sent)
                    raise e
            else:
                tree = sent
            
            # collapse unary fail then continue to discard
            try:
                tree.collapse_unary()
            except:
                # print("Error: ", tree)
                # raise e
                discard += 1
                continue
            # if bad_tree:
            #     tree = fix_empty_leaf_nodes(tree)
            #     tree.collapse_unary()
                
            tree.chomsky_normal_form()
            # flatten 处理：将树平铺为一个句子序列，并添加句末符（EOS）
            in_sentences.append(flatten(tree, add_eos=True))
                
            
            if not isinstance(tree, Tree):
                parses.append(
                    post_process_op(binarize_tree(tree), sp) # FIXED?: SUBWORD TOKENIZATION
                )
            else:
                parses.append(
                    post_process_op(tree, sp)
                )
        
        print("\n# of discarded trees: \n", discard)
        return in_sentences, parses
    dev_in_sentences, dev_parses = parse_sentences(dev_ori)
    test_in_sentences, test_parses = parse_sentences(test_ori)
    train_in_sentences, train_parses = parse_sentences(train_ori)

    

    print("Training set num examples: {}".format(len(train_in_sentences)),
            "Dev set num examples: {}".format(len(dev_in_sentences)),
            "Test set num examples: {}".format(len(test_in_sentences)))

    return train_in_sentences, train_parses, dev_in_sentences, dev_parses, test_in_sentences, test_parses


def make_dataset(
        in_sentences, parses, sp, split="train", with_depth_info=True
):
    
    # 1. tokenize to ids
    def _make_ids_for_split(in_sentences, sp, split="train"):
        ids = [sp.encode_as_ids(sent[:-5]) + [2] for sent in tqdm(in_sentences, desc="Tokenizing {} set".format(split))] # eos added to the end and i don't know if it is correct
        in_len = [len(sent) for sent in ids]
        print(f"{split} set max length: {max(in_len)}")
        return ids, in_len

    ids, in_len = _make_ids_for_split(in_sentences, sp, split)
    data = {
        "ids": ids,
        "lengths": in_len,
        "idxs": list(range(len(in_sentences))),
        # "split": split,
    }
    
    # 2. compute attachment labels
    attach_labels = [
        compute_attachment_labels_text(parse, sp.id_to_piece(id_seq), with_depth_info=with_depth_info)
        for parse, id_seq in tqdm(zip(parses, ids), desc="Computing attachment labels for {}".format(split), total=len(in_sentences))
    ]
    
    data["attachment_labels"] = [
        stack_info_dict["attachment_labels"][1:] for stack_info_dict in attach_labels
    ]
    print("attachment labels done for {} set".format(split))

    # 3. stack tape
    data["stack_tape"] = [
        compute_stack_tape(
            stack_info_dict["attachment_labels"],
            stack_info_dict["cstart_info"],
            type_labels=None,
            with_depth_info=with_depth_info,
        )[:-1, :-1]
        for stack_info_dict in tqdm(attach_labels, desc="Computing stack tape for {}".format(split))
    ]
    if DEBUG:
        print(data["stack_tape"][0].shape)
    print("stack tape done for {} set".format(split))
    
    return data

def chunk_data(data, chunk_size):
    keys = list(data.keys())
    total = len(data[keys[0]])
    chunks = []
    for i in tqdm(range(0, total, chunk_size), desc="Chunking data"):
        chunk = {key: data[key][i:i+chunk_size] for key in keys}
        chunks.append(chunk)
    return chunks

if __name__ == '__main__':
    DEBUG = False
    
    # you should cd data_process and then run this script
    doc_path = "../Dataset/bliip_87_89_wsj/"
    sp=spm.SentencePieceProcessor()
    sp.Load("./spm_parsing/BLLIP_spm.model")
    # debug:
    if DEBUG:
        train_in_sentences, train_parses, dev_in_sentences, dev_parses, test_in_sentences, test_parses=parse_and_split(doc_path, sp, 1, 1, early_stopping=3)
    else:
        train_in_sentences, train_parses, dev_in_sentences, dev_parses, test_in_sentences, test_parses=parse_and_split(doc_path, sp, 1500, 3000)
    chunk_size = 15000
    train_data = make_dataset(train_in_sentences, train_parses, sp, split="train")
    # chunk train_data
    train_chunks = chunk_data(train_data, chunk_size)
    # train_datasets = [HFDataset.from_dict(chunk) for chunk in train_chunks]
    train_datasets = []
    for chunk in tqdm(train_chunks, desc="Recovering train data from chunks"):
        train_datasets.append(HFDataset.from_dict(chunk))
    # train_dataset = HFDataset.from_dict(train_data)
    print("concatenate train datasets")
    train_dataset = concatenate_datasets(train_datasets) 
    train_dataset.save_to_disk("../data/BLLIP_LG_train", max_shard_size="1GB") # LG: large
    del train_data, train_dataset, train_chunks, train_datasets
    print('=' * 100)
    dev_data = make_dataset(dev_in_sentences, dev_parses, sp, split="dev")
    dev_dataset = HFDataset.from_dict(dev_data)
    dev_dataset.save_to_disk("../data/BLLIP_LG_dev", max_shard_size="1GB")
    del dev_data, dev_dataset
    print('=' * 100)
    test_data = make_dataset(test_in_sentences, test_parses, sp, split="test")
    test_dataset = HFDataset.from_dict(test_data)
    test_dataset.save_to_disk("../data/BLLIP_LG_test", max_shard_size="1GB")
    # save to disk with smaller writer batch size
    print('=' * 100)
    
    if DEBUG:
        print("test dataset ids: ", test_dataset[:1]["ids"])
        # ids back to pieces
        tokens = [sp.id_to_piece(id) for id in test_dataset[:1]["ids"][0]]
        print("test dataset tokens: ", tokens)
        print("test dataset lengths: ", test_dataset[:1]["lengths"])
        print("test dataset attachment labels: ", test_dataset[:1]["attachment_labels"])
        print("test dataset attachment labels: ", len(test_dataset[:1]["attachment_labels"][0]))
    print('=' * 100)
    print("vocab size: ", sp.GetPieceSize())