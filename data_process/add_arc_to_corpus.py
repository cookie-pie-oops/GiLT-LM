import json
import logging
import copy
import numpy as np

def tok_with_arc(graph, tok, text, arc=False):
    # graph = [[6],[1,3]] means 6->1 and 1->2 and 3->2, index from 1
    # dm and psd index from 0
    word_piece_tok = text.split(" ")
    graph_tok = copy.deepcopy(word_piece_tok)
    span_dict = {}

    #compose "." with number or name arround it. They don't need to be separated.
    complete_index = 0
    for i, word in enumerate(word_piece_tok):
        range_down_index = copy.deepcopy(complete_index)
        for j in range(range_down_index, len(tok)):
            if word == tok[j]:
                span_dict[j + 1] = i + 1
                complete_index += 1
                break
            # match word in tok[j] and match the next
            elif tok[j] in word:
                span_dict[j + 1] = i + 1
                complete_index += 1
                word = word[len(tok[j]):].strip()
            elif word in tok[j]:
                span_dict[j + 1] = i + 1 
                tok[j] = tok[j][len(word):].strip()
                break
            if word == "":
                break

    graph_matrix = np.zeros((len(word_piece_tok) + 1, len(word_piece_tok) + 1))
    for (tail_index, head_index_list) in enumerate(graph):
        for head_index in head_index_list:
            if tail_index + 1 not in span_dict:
                import pdb; pdb.set_trace()
            if head_index[0] not in span_dict:
                if head_index[0] != 0:
                    import pdb; pdb.set_trace()
                else: #root
                    graph_matrix[head_index[0], span_dict[tail_index + 1]] = 1
            else:
                graph_matrix[span_dict[head_index[0]], span_dict[tail_index + 1]] = 1 #index start from 1
    
    # add arc to tok list in row i and col i
    arc_num = 0
    for i in range(len(word_piece_tok)):
        row = graph_matrix[i, 0:i]
        col = graph_matrix[0:i, i]

        if arc:
            for index, arc_exist in enumerate(row): #(i -> i-j)
                if arc_exist == 1:
                    if index == 0:
                        graph_tok.insert(i + arc_num, "-> root")
                        logging.warning("root shell not be pointed.")
                    else:
                        graph_tok.insert(i + arc_num, "-> " + word_piece_tok[index - 1])
                    arc_num += 1
            
            for index, arc_exist in enumerate(col): #(i-j -> i)
                if arc_exist == 1:
                    if index == 0:
                        graph_tok.insert(i + arc_num, "<- root")
                    else:
                        graph_tok.insert(i + arc_num, "<- " + word_piece_tok[index - 1])
                    arc_num += 1
        else:
            for index, arc_exist in enumerate(row): #(i -> i-j)
                if arc_exist == 1:
                    if index == 0:
                        graph_tok.insert(i + arc_num, "-> ->2") #0
                        logging.warning("root shell not be pointed.")
                    else:
                        graph_tok.insert(i + arc_num, "-> ->2") #index - 1
                    arc_num += 1
            
            for index, arc_exist in enumerate(col): #(i-j -> i)
                if arc_exist == 1:
                    if index == 0:
                        graph_tok.insert(i + arc_num, "<- <-2")
                    else:
                        graph_tok.insert(i + arc_num, "<- <-2")
                    arc_num += 1
    
    return " ".join(graph_tok)

def update_arc_to_corpus(file_list):
    dataset_name = "transition_sequence"
    for f in file_list:
        fp = open(f"./parser/BLLIP_LG_{f}_multi.json")
        fw1 = open(f"./{dataset_name}/BLLIP_LG_{f}_pas.txt", "w")
        fw2 = open(f"./{dataset_name}/BLLIP_LG_{f}_psd.txt", "w")
        fw3 = open(f"./{dataset_name}/BLLIP_LG_{f}_dm.txt", "w")
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            if len(data["tok"]) != len(data["pas_graph"]):
                logging.warning("Length inequility.")

            pas_sequence = tok_with_arc(data["pas_graph"], copy.deepcopy(data["tok"]), data["text"], True)

            psd_sequence = tok_with_arc(data["psd_graph"], copy.deepcopy(data["tok"]), data["text"], True)

            dm_sequence = tok_with_arc(data["dm_graph"], copy.deepcopy(data["tok"]), data["text"], True)

            fw1.write(pas_sequence+'\n')
            fw2.write(psd_sequence+'\n')
            fw3.write(dm_sequence+'\n')

        fp.close()
           
    
if __name__ == "__main__":
    file_list = ["DEV", 'TEST', "TRAIN"]
    update_arc_to_corpus(file_list)
        
    
    