import json
import numpy as np

if __name__ == '__main__':
    file_path = "./DTG_data/bllip_train_LG_standard_rbt.txt"
    raw_file_path = "./BLLIP_LG_TRAIN.txt"
    fw = open("./DTG_data/dtg_train_multiarrow.txt", 'w')
    raw_texts = open(raw_file_path, 'r').readlines()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line, raw_text in zip(lines, raw_texts):
            word_list = line.strip().split(' ')
            raw_text_list = raw_text.strip().split(' ')
            word_stack = []
            adj_matrix = np.zeros((len(raw_text_list) + 1, len(raw_text_list) + 1))
            id = 1
            left_arc_list = []
            right_arc_list = []

            for word in word_list:
                if word != 'left_arc' and word != 'right_arc' and word != 'pop_root':
                    word_stack.append({'id': id, 'word': word})
                    id += 1
                elif word == 'left_arc':
                    top1 = word_stack.pop()
                    top2 = word_stack.pop()
                    adj_matrix[top1['id'], top2['id']] = 1
                    word_stack.append(top1)
                elif word == 'right_arc':
                    top1 = word_stack.pop()
                    top2 = word_stack.pop()
                    adj_matrix[top2['id'], top1['id']] = 1
                    word_stack.append(top2)
                elif word == 'pop_root':
                    top = word_stack.pop()
                    adj_matrix[0, top['id']] = 1
            assert len(word_stack) == 0
            
            for i in range(len(raw_text_list)):
                row = adj_matrix[i+1, 0:i+2]
                col = adj_matrix[0:i+2, i+1]

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

            fw.write(json.dumps(arrow_dict, ensure_ascii=False)+'\n')
