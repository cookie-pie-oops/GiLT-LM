import os

def parse_dependency_tree(tree_str):
    pieces = tree_str.split(" ")
    sentence = []
    pre_piece = ""
    for piece in pieces:
        if piece == "":
            continue
        if piece[0] != '(':
            if pre_piece != "" and "NONE" in pre_piece:
                continue
            sentence.append(piece.strip(")"))
        pre_piece = piece
    
    if "*U*" in sentence:
        sentence.remove("*U*")
    return ' '.join(sentence)

def parse_whole_section(doc_path, dataset_list, field=""):
    # the sentence number is restricted for TEST and DEV
    count_num = 0
    restrict = {"TEST":100000, "DEV":50000}

    for files in os.listdir(doc_path):
        if field != "" and count_num >= restrict[field]:
            break
        file = os.path.join(doc_path, files)
        with open(file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            full_content = "".join([line.replace("\t"," ").strip("\n") for line in lines])
            origin_sentences = full_content.split("(S1")

            for origin_sentence in origin_sentences:
                if origin_sentence == "":
                    continue
                dependency_tree = origin_sentence
                sentence = parse_dependency_tree(dependency_tree)
                count_num += 1
                dataset_list.append(sentence)
                if field != "" and count_num >= restrict[field]:
                    break
            f.close()
    return dataset_list

def parse_and_split():
    years = [1987, 1988, 1989]

    BLIIP_LG = []
    BLIIP_LG_DEV = []
    BLIIP_LG_TEST = []

    for year in years:
        dev_flag = 0
        test_flag = 0
        year_data = f"../Dataset/bliip_87_89_wsj/{year}"
        for file_index in os.listdir(year_data):
            # for each year, the first doc is for dev
            if dev_flag == 0:
                BLIIP_LG_DEV = parse_whole_section(os.path.join(year_data,file_index), BLIIP_LG_DEV, "DEV")
                dev_flag = 1
            # for each year, the second doc is for test
            elif test_flag == 0:
                BLIIP_LG_TEST = parse_whole_section(os.path.join(year_data,file_index), BLIIP_LG_TEST, "TEST")
                test_flag = 1
            else:
                BLIIP_LG = parse_whole_section(os.path.join(year_data,file_index), BLIIP_LG)
    
    train_file = open("./BLLIP_LG_TRAIN.txt",'w',encoding='utf-8')
    train_file.write("\n".join(BLIIP_LG))
    dev_file = open("./BLLIP_LG_DEV.txt",'w',encoding='utf-8')
    dev_file.write("\n".join(BLIIP_LG_DEV))
    test_file = open("./BLLIP_LG_TEST.txt",'w',encoding='utf-8')
    test_file.write("\n".join(BLIIP_LG_TEST))


def clean_test_from_train():
    import json
    #delete polluted data in train set
    train_fp = open('./parser/backup_train_multi.json', "r")
    output_train_json = open('./parser/BLLIP_LG_TRAIN_multi_clean.json', 'w')

    years = [1987, 1988, 1989]

    BLIIP_LG_DEV = []
    BLIIP_LG_TEST = []

    for year in years:
        dev_flag = 0
        test_flag = 0
        year_data = f"../Dataset/bliip_87_89_wsj/{year}"
        for file_index in os.listdir(year_data):
            # for each year, the first doc is for dev
            if dev_flag == 0:
                BLIIP_LG_DEV = parse_whole_section(os.path.join(year_data,file_index), BLIIP_LG_DEV, "DEV")
                dev_flag = 1
            # for each year, the second doc is for test
            elif test_flag == 0:
                BLIIP_LG_TEST = parse_whole_section(os.path.join(year_data,file_index), BLIIP_LG_TEST, "TEST")
                test_flag = 1
    
    polluted_train_list = train_fp.readlines()
    for polluted_train_data in polluted_train_list:
        data = json.loads(polluted_train_data)
        if data["text"] in BLIIP_LG_DEV:
            BLIIP_LG_DEV.remove(data["text"])
        elif data["text"] in BLIIP_LG_TEST:
            BLIIP_LG_TEST.remove(data["text"])
        else:
            json_str = json.dumps(data, ensure_ascii=False)
            output_train_json.write(json_str + "\n")


def view_corpus():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    from collections import Counter
    lens_distribution = []
    with open("./BLLIP_LG.txt",'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 3:
                print(line.strip())
            lens_distribution.append(len(line.strip()))
        print(f"max length is {max(lens_distribution)}")
        f.close()

    length_counts = Counter(lens_distribution)
    lengths = list(length_counts.keys())
    frequencies = list(length_counts.values())
    plt.bar(lengths, frequencies, color='skyblue')
    plt.xlabel("Length of Sentence")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sentence Lengths in the Corpus")
    plt.show()

def pos_word_group(pos_list, word_list):
    res = []
    for (pos, word) in zip(pos_list, word_list):
        res.append((word, pos))
    return res

def hanlp_parse():
    import hanlp
    import json
    import tqdm
    # psd_parser = hanlp.load('SEMEVAL15_PSD_BIAFFINE_EN')
    # pas_parser = hanlp.load('SEMEVAL15_PAS_BIAFFINE_EN')
    # dm_parser = hanlp.load('SEMEVAL15_DM_BIAFFINE_EN')

    # tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    Hanlp_Parser = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)

    for name in ["DEV", "TEST"]:
        with open("./BLLIP_LG_" + name + ".txt",'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines):
                json_data = {"text":line.strip()}

                pas_graph = Hanlp_Parser(line.strip(), tasks='sdp/pas')
                tok = line.strip().split(" ")
                json_data["tok"] = pas_graph["tok"]
                json_data["pas_graph"] = pas_graph["sdp/pas"]

                dm_graph = Hanlp_Parser(line.strip(), tasks='sdp/dm')
                json_data["dm_graph"] = dm_graph["sdp/dm"]    

                # pos = tagger(tok)
                # psd_graph = psd_parser(pos_word_group(pos, tok))
                # pas_graph = pas_parser(pos_word_group(pos, tok))
                # dm_graph = dm_parser(pos_word_group(pos, tok))


                psd_graph = Hanlp_Parser(line.strip(), tasks='sdp/psd')
                json_data["psd_graph"] = psd_graph["sdp/psd"]
                # json_data["psd_graph"] = psd_graph
                # json_data["pas_graph"] = pas_graph
                # json_data["dm_graph"] = dm_graph

                json_str = json.dumps(json_data, ensure_ascii=False)
                with open("./BLLIP_LG_" + name + "_single_task.json",'a',encoding='utf-8') as wf:
                    wf.write(json_str + "\n")
            f.close()

def arc_num_of_graph(graph):
    arc_num = 0
    for node in graph:
        arc_num += len(node)
    return arc_num

def view_json():
    import json
    pas_arc_num = 0
    dm_arc_num = 0
    psd_arc_num = 0

    max_pas_arc_num = 0
    max_dm_arc_num = 0
    max_psd_arc_num = 0

    line_num = 0
    with open("./BLLIP_LG_TRAIN_multi.json",'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_num += 1

            data = json.loads(line)
            
            pas_arc_num += arc_num_of_graph(data['pas_graph'])
            dm_arc_num += arc_num_of_graph(data['dm_graph'])
            psd_arc_num += arc_num_of_graph(data['psd_graph'])

            max_pas_arc_num = max(max_pas_arc_num, arc_num_of_graph(data['pas_graph']))
            max_dm_arc_num = max(max_dm_arc_num, arc_num_of_graph(data['dm_graph']))
            max_psd_arc_num = max(max_psd_arc_num, arc_num_of_graph(data['psd_graph']))
    print(f"avg pas arcs num :{pas_arc_num / line_num}, avg dm arcs num :{dm_arc_num / line_num}, avg psd arcs num :{psd_arc_num / line_num}")
    print(f"max pas arcs num :{max_pas_arc_num}, max dm arcs num :{max_dm_arc_num}, max psd arcs num :{max_psd_arc_num}")


if __name__=="__main__":
    clean_test_from_train()

