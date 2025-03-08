import os

#python train.py --config config/psd_parsing_model.yaml --parse --target_dir /home/huangty/.flair/datasets/enhanced_ud/PSD --keep_order

def convert_test(file_path, target_dir):
    with open(file_path, "r") as f:
        lines = f.readlines()

    target_file_path = f"{target_dir}/test/train.tsv"
    if not os.path.exists(os.path.dirname(target_file_path)):
        os.makedirs(os.path.dirname(target_file_path))
        os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/test.tsv {target_dir}/test/test.tsv")
        os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/dev.tsv {target_dir}/test/dev.tsv")

    write_file = open(target_file_path, "w")
    for line in lines:
        line_list = line.strip(" ").split()
        line_length = len(line_list)
        for i in range(1, line_length + 1):
            word = line_list[i - 1]
            convert_str = f"{i}\t{word}\tX\tX\tX\t_\t0\t_\t_"
            write_file.write(convert_str + "\n")
        write_file.write("\n")


def convert_dev(file_path, target_dir):
    with open(file_path, "r") as f:
        lines = f.readlines()

    target_file_path = f"{target_dir}/dev/train.tsv"
    if not os.path.exists(os.path.dirname(target_file_path)):
        os.makedirs(os.path.dirname(target_file_path))
        os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/test.tsv {target_dir}/dev/test.tsv")
        os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/dev.tsv {target_dir}/dev/dev.tsv")

    write_file = open(target_file_path, "w")
    for line in lines:
        line_list = line.strip(" ").split()
        line_length = len(line_list)
        for i in range(1, line_length + 1):
            word = line_list[i - 1]
            convert_str = f"{i}\t{word}\tX\tX\tX\t_\t0\t_\t_"
            write_file.write(convert_str + "\n")
        write_file.write("\n")


def convert_train_with_seperate(file_path, target_dir):
    with open(file_path, "r") as f:
        lines = f.readlines()

    seperate_num = 18
    for i in range(seperate_num):
        seperate_lines = lines[int((len(lines)/seperate_num)*i): int((len(lines)/seperate_num)*(i+1))]
        target_file_path = f"{target_dir}/train_{i + 1}/train.tsv"
        if not os.path.exists(os.path.dirname(target_file_path)):
            os.makedirs(os.path.dirname(target_file_path))
            os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/test.tsv {target_dir}/train_{i + 1}/test.tsv")
            os.system(f"cp /home/huangty/.flair/datasets/enhanced_ud/PSD/dev.tsv {target_dir}/train_{i + 1}/dev.tsv")

        write_file = open(target_file_path, "w")
        for line in seperate_lines:
            line_list = line.strip(" ").split()
            line_length = len(line_list)
            for i in range(1, line_length + 1):
                word = line_list[i - 1]
                convert_str = f"{i}\t{word}\tX\tX\tX\t_\t0\t_\t_"
                write_file.write(convert_str + "\n")
            write_file.write("\n")

if __name__ == '__main__':
    datasets = ["DEV", "TEST", "TRAIN"]
    sdp_name = "psd"
    target_dir = "/home/huangty/.flair/datasets/enhanced_ud/BLLIP_LG"

    # test and dev don't need seperate
    convert_test("./BLLIP_LG_TEST.txt", target_dir)
    convert_dev("./BLLIP_LG_DEV.txt", target_dir)

    convert_train_with_seperate("./BLLIP_LG_TRAIN.txt", target_dir)