import json
import logging
import copy
import numpy as np

def tok_with_arc(graph, tok, text, arc=False):
    # graph = [[6],[1,3]] means 6->1 and 1->2 and 3->2, index from 1
    # dm and psd index from 0
    # word_piece_tok = text.split(" ")
    graph_tok = copy.deepcopy(tok)
    arrow_list = []
    left_arc_list = []
    right_arc_list = []
    # span_dict = {}

    #compose "." with number or name arround it. They don't need to be separated.
    # complete_index = 0
    # for i, word in enumerate(word_piece_tok):
    #     range_down_index = copy.deepcopy(complete_index)
    #     for j in range(range_down_index, len(tok)):
    #         if word == tok[j]:
    #             span_dict[j + 1] = i + 1
    #             complete_index += 1
    #             break
    #         # match word in tok[j] and match the next
    #         elif tok[j] in word:
    #             span_dict[j + 1] = i + 1
    #             complete_index += 1
    #             word = word[len(tok[j]):].strip()
    #         elif word in tok[j]:
    #             span_dict[j + 1] = i + 1 
    #             tok[j] = tok[j][len(word):].strip()
    #             break
    #         if word == "":
    #             break

    graph_matrix = np.zeros((len(tok) + 1, len(tok) + 1)) # there exist root means 0
    for node_info in graph:
        for head_id in node_info["head"]:
            graph_matrix[head_id, int(node_info["id"])] = 1
            # if tail_index + 1 not in span_dict:
            #     import pdb; pdb.set_trace()
            # if head_index[0] not in span_dict:
            #     if head_index[0] != 0:
            #         import pdb; pdb.set_trace()
            #     else: #root
            #         graph_matrix[head_index[0], span_dict[tail_index + 1]] = 1
            # else:
            #     graph_matrix[span_dict[head_index[0]], span_dict[tail_index + 1]] = 1 #index start from 1
    
    # add arc to tok list in row i and col i
    arc_num = 0
    for i in range(len(tok)):
        row = graph_matrix[i+1, 0:i+2]
        col = graph_matrix[0:i+2, i+1]

        if arc:
            for index, arc_exist in enumerate(row): #(i -> j)
                if arc_exist == 1 and index != 0:
                    graph_tok.insert(i + arc_num, "-> " + tok[index - 1])
                    arc_num += 1
            
            for index, arc_exist in enumerate(col): #(j -> i)
                if arc_exist == 1 and index != 0:
                    graph_tok.insert(i + arc_num, "<- " + tok[index - 1])
                    arc_num += 1
        else:
            for index, arc_exist in enumerate(row): #(i -> j)
                if arc_exist == 1 and index != 0:
                    graph_tok.insert(i + arc_num, "-> ->2") #0
                    arrow_list.append(str(index)) #index start from 1
                    arc_num += 1
            
            for index, arc_exist in enumerate(col): #(j -> i)
                if arc_exist == 1 and index != 0:
                    graph_tok.insert(i + arc_num, "<- <-2")
                    arrow_list.append(str(index)) #index start from 1
                    arc_num += 1

        # multi-arc prediction
        temp_token_left_list = []
        temp_token_right_list = []
        for index, arc_exist in enumerate(row): # i -> j
            if arc_exist == 1:
                temp_token_right_list.append(int(index)) #index start from 1
                
        for index, arc_exist in enumerate(col): # j -> i
            if arc_exist == 1:
                temp_token_left_list.append(int(index)) #index start from 1

        left_arc_list.append(temp_token_left_list)
        right_arc_list.append(temp_token_right_list)
    arrow_dict = {"left_arc_list":left_arc_list,"right_arc_list":right_arc_list}
    # return " ".join(graph_tok), ",".join(arrow_list)
    return arrow_dict

def update_arc_to_corpus(file_list):
    Dataset_name = "sampling_parse_graph"
    # pay attention to ->2 and <-2!!!
    if "arc" in Dataset_name:
        arc_name = True
    else:
        arc_name = False
    num = 300
    for f in file_list:
        # fp = open(f"./parser/BLLIP_LG_{f}_biaffine.json")
        # fp = open(f"./parser/{f}_ACE.json")
        fp = open(f"./sampling_parse_graph/{f}_ACE_sampling_parse_graph_psd_{num}_2.json")
        # fw1 = open(f"./{Dataset_name}/BLLIP_LG_{f}_pas_{num}.txt", "w")
        # fw2 = open(f"./{Dataset_name}/BLLIP_LG_{f}_psd_{num}.txt", "w")
        # fw3 = open(f"./{Dataset_name}/{f}_dm_Supar.txt", "w")
        lines = fp.readlines()

        # fw4  = open(f"./{Dataset_name}/BLLIP_LG_{f}_pas_multiarrow.txt", "w") # for pointer network
        fw5  = open(f"./{Dataset_name}/{f}_psd_ACE_multiarrow_sampling_parse_graph.txt", "w")
        # fw6  = open(f"./{Dataset_name}/{f}_dm_Supar_multiarrow.txt", "w")
        for line in lines:
            data = json.loads(line)

            # pas_sequence, arrow_list1 = tok_with_arc(data["pas_graph"], data["text"].split(" "), data["text"], arc_name)
            # arrow_dict1 = tok_with_arc(data["pas_graph"], data["text"].split(" "), data["text"], arc_name)

            # psd_sequence, arrow_list2 = tok_with_arc(data["psd_graph"], data["text"].split(" "), data["text"], arc_name)
            arrow_dict2 = tok_with_arc(data["psd_graph"], data["text"].split(" "), data["text"], arc_name)

            # dm_sequence, arrow_list3 = tok_with_arc(data["dm_graph"], data["text"].split(" "), data["text"], arc_name)
            # arrow_dict3 = tok_with_arc(data["dm_graph"], data["text"].split(" "), data["text"], arc_name)

            # fw1.write(pas_sequence+'\n')
            # fw2.write(psd_sequence+'\n')
            # fw3.write(dm_sequence+'\n')

            # fw4.write(json.dumps(arrow_dict1, ensure_ascii=False)+'\n')
            fw5.write(json.dumps(arrow_dict2, ensure_ascii=False)+'\n')
            # fw6.write(json.dumps(arrow_dict3, ensure_ascii=False)+'\n')

        fp.close()
           
    
if __name__ == "__main__":
    file_list = ['TEST','DEV']
    # file_list = ['TEST']
    update_arc_to_corpus(file_list)
        
    
    