import json

def list_add(a, b):
    new_a = []
    for item in a:
        if item != 0:
            item += b
        new_a.append(item)
    return new_a

if __name__ == "__main__":
    finetune_set = "STS"
    data_splits = ["TRAIN", "DEV", "TEST"]
    sdp_name = "psd"
    for data_split in data_splits:
        file_name = f"./{finetune_set}/{finetune_set}_{data_split}_{sdp_name}_multiarrow.txt"
        write_file_name = f"./{finetune_set}/{finetune_set}_{data_split}_{sdp_name}_multiarrow_2.txt"
        fw = open(write_file_name, "w")
        with open(file_name, "r") as f:
            lines = f.readlines()
            for i in range(len(lines) // 2):
                sentence1_arrow = json.loads(lines[i * 2].strip())
                sentence2_arrow = json.loads(lines[i * 2 + 1].strip())
                sentence1_wrods_account = len(sentence1_arrow["left_arc_list"])
                for left_arc in sentence2_arrow["left_arc_list"]:
                    sentence1_arrow["left_arc_list"].append(list_add(left_arc, sentence1_wrods_account))
                for right_arc in sentence2_arrow["right_arc_list"]:
                    sentence1_arrow["right_arc_list"].append(list_add(right_arc, sentence1_wrods_account))
                fw.write(json.dumps(sentence1_arrow) + "\n")
        fw.close()
