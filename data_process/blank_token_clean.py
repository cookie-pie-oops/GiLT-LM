import copy


def main():
    # delete the blank before the my special tokens <- ->
    # split = ['TRAIN', 'DEV', 'TEST']
    split = ['TEST']
    # sdp = ['dm', 'pas', 'psd']
    sdp = ["psd"]
    dataset_name = "sdp_trans_silver_parse_graph"
    
    token_list = []
    spm_vocab_path = './spm_parsing/BLLIP_spm.vocab'
    with open(spm_vocab_path, 'r') as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            token, _ = line.rstrip().split("\t")
            if token in ["<-", "->", "<-2", "->2"]:
                token_list.append(i)
            if token == "▁":
                whiteblank = i

    num = 1800
    for name1 in split:
        for name2 in sdp:
            remove_num = 0
            arc_num = 0

            arc_with_blank_file_path = f'./{dataset_name}/BLLIP_LG_{name1}_SPM_{name2}_{num}.txt'
            f2 = open(f'./{dataset_name}/BLLIP_LG_{name1}_{name2}_{num}.csv', 'w')
            with open(arc_with_blank_file_path, 'r') as f:
                lines = f.readlines()
                sentence_num = len(lines)
                for line in lines:
                    line = line.strip().split(" ")
                    clean_list = copy.deepcopy(line)
                    del_num = 1
                    for i in range(len(line)):
                        if  i != 0 and int(line[i]) in token_list:
                            del clean_list[i - del_num]
                            del_num += 1
                            remove_num += 1
                            arc_num += 1
                    
                    f2.write(",".join(clean_list) + "\n")
    
            print(remove_num, arc_num)
            print(arc_num/sentence_num)


if __name__ == '__main__':
    main()