import os
from tqdm import tqdm
from nltk.tree import Tree
from data_utils import flatten, binarize_tree, post_process_op
import sentencepiece as spm
from attachment_utils import compute_attachment_labels_text
from stack_tape_utils import compute_stack_tape
# HFDataset
from datasets import Dataset as HFDataset
DEBUG = True

def _list_create(doc_path, early_stopping=-1):  # -1 for no limit
    dataset_list = []
    years = ['1987', '1988', '1989']
    cnt = 0
    if early_stopping < -1:
        raise ValueError(
            "early_stopping should be -1 or a nonnegative integer")
    for year in os.listdir(doc_path):
        if year not in years:
            continue
        year_dir = os.path.join(doc_path, year)
        if not os.path.isdir(year_dir):
            continue
        # 遍历每个子目录（dir_index）
        for sub_dir in tqdm(os.listdir(year_dir), desc=f"Processing year {year}"):
            sub_dir_path = os.path.join(year_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            # print(f"  Processing subdir: {sub_dir}")
            for file_name in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    full_content = "".join(
                        [line.replace("\t", " ").strip("\n") for line in lines])
                    # 以"(S1"作为分割标记
                    origin_sentences = full_content.split("(S1")
                    for origin_sentence in origin_sentences:
                        if cnt >= early_stopping and early_stopping != -1:
                            return dataset_list
                        origin_sentence = origin_sentence.strip()
                        if origin_sentence.strip() == "":
                            continue
                        complete_sentence = "(S1" + origin_sentence
                        # 在这里进行解析，例如调用 constituency_parse 函数
                        dataset_list.append(complete_sentence)
                        cnt += 1

    return dataset_list


def parse_and_split(doc_path, sp, dev_num=50000, test_num=100000, early_stopping=-1):

    # doc_path = "../Dataset/bliip_87_89_wsj/"
    original_list = _list_create(doc_path, early_stopping=early_stopping)
    # split to train,dev,test
    # shuffle
    import random
    # seed
    random.seed(12345)
    random.shuffle(original_list)
    # split
    train_ori = original_list[dev_num + test_num:]
    dev_ori = original_list[:dev_num]
    test_ori = original_list[dev_num:dev_num + test_num]
    
    def fix_empty_labels(t):
        if isinstance(t, str):
            return t
        # t is a tree
        if not t.label().strip():
            t.set_label("<unk>")
        # empty children
        for child in t:
            fix_empty_labels(child)
        return t

    def parse_sentences(ori_data):
        discarded = 0
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
            
            
            # 如果转换后的 tree 仍不是 Tree（理论上不应该出现），则调用 binarize_tree
            tree = fix_empty_labels(tree)
            try:
                tree.collapse_unary()
            except:
                # print("Error: ", tree)
                # raise e
                discarded += 1
                continue
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
        return in_sentences, parses

    train_in_sentences, train_parses = parse_sentences(train_ori)
    dev_in_sentences, dev_parses = parse_sentences(dev_ori)
    test_in_sentences, test_parses = parse_sentences(test_ori)

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
        train_in_sentences, train_parses, dev_in_sentences, dev_parses, test_in_sentences, test_parses=parse_and_split(doc_path, sp, 50000, 100000)
    train_data = make_dataset(train_in_sentences, train_parses, sp, split="train")
    print('=' * 100)
    dev_data = make_dataset(dev_in_sentences, dev_parses, sp, split="dev")
    print('=' * 100)
    test_data = make_dataset(test_in_sentences, test_parses, sp, split="test")
    # make HF dataset
    train_dataset = HFDataset.from_dict(train_data)
    dev_dataset = HFDataset.from_dict(dev_data)
    test_dataset = HFDataset.from_dict(test_data)
    # save to disk
    print('=' * 100)
    train_dataset.save_to_disk("../data/BLLIP_LG_train") # LG: large
    dev_dataset.save_to_disk("../data/BLLIP_LG_dev")
    test_dataset.save_to_disk("../data/BLLIP_LG_test")
    # first 10 of test for testing
    print('=' * 100)
    if DEBUG:
        print("test dataset ids: ", test_dataset[:1]["ids"])
        # ids back to pieces
        tokens = [sp.id_to_piece(id) for id in test_dataset[:1]["ids"][0]]
        print("test dataset tokens: ", tokens)
        print("test dataset lengths: ", test_dataset[:1]["lengths"])
        print("test dataset attachment labels: ", test_dataset[:1]["attachment_labels"])
        print("test dataset attachment labels: ", len(test_dataset[:1]["attachment_labels"][0]))
    
    print("vocab size: ", sp.GetPieceSize())