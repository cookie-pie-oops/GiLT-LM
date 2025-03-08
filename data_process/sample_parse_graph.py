import ast
import hanlp
import heapq
import math
import json
import time
import tqdm
import copy

def pos_word_group(pos_list, word_list):
    # (word, pos)
    res = []
    for i in range(len(pos_list)):
        sentence_res = []
        for (pos, word) in zip(pos_list[i], word_list[i]):
            sentence_res.append((word, pos))
        res.append(sentence_res)
    return res

def gain_score():
    #/.conda/envs/DTG/lib/python3.10/site-packages/hanlp/components/parsers/biaffine_tf/model.py 128 lines
    psd_parser = hanlp.load('SEMEVAL15_DM_BIAFFINE_EN')
    tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    with open("./BLLIP_LG_" + "TEST" + ".txt",'r',encoding='utf-8') as f:
        lines = f.readlines()
        batch_size = 256
        batched_line = []
        for line in tqdm.tqdm(lines):
            if len(batched_line) < batch_size:
                batched_line.append(line.strip().split(" "))
            else:
                tag = tagger(batched_line)
                psd_graph = psd_parser(pos_word_group(tag, batched_line))
                batched_line = [line.strip().split(" ")]
                import pdb;pdb.set_trace()
        
        if batched_line:
            tag = tagger(batched_line)
            psd_graph = psd_parser(pos_word_group(tag, batched_line))

def get_possible_parse_graph(score, tok, num):
    # top k parse graph
    min_heap = []
    for i in range(len(tok) + 1):
        for j in range(len(tok) + 1):
            if not min_heap:
                if not math.isinf(score[i][j]):
                    heapq.heappush(min_heap, (score[i][j], "1"))
                    heapq.heappush(min_heap, (0, "0"))
                else:
                    heapq.heappush(min_heap, (0, "0"))
            else:
                next_step_min_heap = []
                for k in range(len(min_heap)):
                    (sum_score, arc_select) = min_heap[k]

                    choose_seq = arc_select + "1"
                    not_choose_seq = arc_select + "0"

                    if i != j:
                        if len(next_step_min_heap) <= num:
                            if not math.isinf(score[i][j]):
                                heapq.heappush(next_step_min_heap, (sum_score + score[i][j], choose_seq))
                                heapq.heappush(next_step_min_heap, (sum_score, not_choose_seq))
                            else:
                                heapq.heappush(next_step_min_heap, (sum_score, not_choose_seq))
                        else:
                            if not math.isinf(score[i][j]):
                                if sum_score + score[i][j] > min_heap[-1][0]:
                                    heapq.heappushpop(next_step_min_heap, (sum_score + score[i][j], choose_seq))
                                else:
                                    heapq.heappushpop(next_step_min_heap, (sum_score, not_choose_seq))
                            else:
                                heapq.heappushpop(next_step_min_heap, (sum_score, not_choose_seq))    
                    else:
                        heapq.heappush(next_step_min_heap, (sum_score, not_choose_seq))       
                    
                min_heap = heapq.nlargest(num, next_step_min_heap)

    return min_heap, len(score)

def dfs(node, graph, visited, rec_stack):
    visited[node] = True
    rec_stack[node] = True
    
    for next_node, has_edge in enumerate(graph[node]):
        if has_edge:
            if not visited[next_node]:
                if dfs(next_node, graph, visited, rec_stack):
                    return True
            elif rec_stack[next_node]:
                return True
    
    rec_stack[node] = False
    return False

def is_cyclic(graph):
    V = len(graph)
    visited = [False] * V
    rec_stack = [False] * V
    cycle_count = 0

    for i in range(V):
        if not visited[i]:
            if dfs(i, graph, visited, rec_stack):
                cycle_count += 1
    return 1 if cycle_count > 0 else 0  

def write_into_file(parse_graph, text, tok, padding_length, sdp_name, dataset, nums):
    # length not equal to len(parse_str) due to the padding, but don't make a difference
    if len(parse_graph) <= nums:
        parse_graph.extend([parse_graph[-1]] * (nums - len(parse_graph)))

    assert len(parse_graph) == nums
    sum_cycles = 0
    roots_not_one_child = 0
    roots_have_parent = 0
    for possible_parse in parse_graph:
        (score, parse_str) = possible_parse
        assert len(parse_str) == (len(tok) + 1)**2

        json_obj = {"text": text}
        sdp_graph = []

        adj_matrix = []
        for i in range(len(tok) + 1):
            adj_matrix.append([0]*(len(tok) + 1))

        for row_num in range(len(tok) + 1):
            dependency_arc = parse_str[(len(tok) + 1)*row_num:(len(tok) + 1)*(row_num + 1)]
            head_list = []
            for i, tag in enumerate(dependency_arc):
                if tag == "1":      # i -> row_num
                    head_list.append(i)
                    adj_matrix[i][row_num] = 1
                elif tag == "0":
                    continue

            if row_num != 0:
                sdp_graph.append({
                    "id": row_num,
                    "form": tok[row_num - 1],
                    "head": head_list,
                    })
        json_obj[f"{sdp_name}_graph"] = sdp_graph

        silver_parse_graph_file = open(f"./sampling_parse_graph/{dataset}_ACE_sampling_parse_graph_{sdp_name}_{nums}_2.json", "a")
        silver_parse_graph_file.write(json.dumps(json_obj) + "\n")

        sum_cycles += is_cyclic(adj_matrix)
        # root have more than one child or root have father
        if sum(1 for element in adj_matrix[0] if element == 1) != 1:
            roots_not_one_child += 1
        if sum([line[0] for line in adj_matrix]) != 0:
            roots_have_parent += 1          
    
    return sum_cycles / nums, roots_not_one_child / nums, roots_have_parent / nums


def gen_silver_parse_graph():
    # make file
    Datasets = ["DEV", "TEST"]
    SDP_NAME = ["psd", "pas", "dm"]
    SDP_NAME = ["psd"]
    num_list = [300]
    for num in num_list:
        for sdp_name in SDP_NAME:
            cycles = 0
            roots_not_one_child_nums = 0
            roots_have_parent_nums = 0
            for dataset in Datasets:
                score_file = open(f"./sampling_parse_graph/{dataset}_ACE_score_{sdp_name}.txt", "r")
                socres = score_file.readlines()
                txt_file = open(f"./BLLIP_LG_{dataset}.txt", "r")
                texts = txt_file.readlines()

                assert len(socres) == len(texts), "the lengths of scores and texts do not equal"
                for score, text in zip(socres, texts):
                    tok = text.strip().split(" ")
                    graph_size = len(tok) + 1  # root
                    score_list = json.loads(score)["score"]
                    score_list = score_list[:graph_size]
                    for i in range(len(score_list)):
                        score_list[i] = score_list[i][:graph_size]
                    
                    # moderate
                    min_value = min([min(row) for row in score_list])
                    score_list[0] = [min_value - 1] * graph_size
                    # first_column = [row[0] for row in score_list]
                    # max_in_first_column = max(first_column)
                    # for i in range(len(score_list)):
                    #     if score_list[i][0] != max_in_first_column:
                    #         score_list[i][0] = min_value

                    silver_parses, padding_len = get_possible_parse_graph(score_list, tok, num)
                    cycle_num, roots_not_one_child, roots_have_parent = write_into_file(silver_parses, text.strip(), tok, padding_len, sdp_name, dataset, num)
                    cycles += cycle_num
                    roots_not_one_child_nums += roots_not_one_child
                    roots_have_parent_nums += roots_have_parent

                print(sdp_name, "mean cycles : ", cycles / len(texts))
                print(sdp_name, "roots not one child : ", roots_not_one_child_nums / len(texts))
                print(sdp_name, "roots have parent : ", roots_have_parent_nums / len(texts))
                print(len(texts))

if __name__ == '__main__':
    # gain_score()
    gen_silver_parse_graph()