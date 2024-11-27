
def compete_token_num(token_level_file, arc_point_file, sdp):
    print(f"  {sdp} data distrubution:")
    token_level_sum = 0
    arc_level_sum = 0
    extra_token_sum = 0

    f2 = open(arc_point_file, "r")
    with open(token_level_file, "r") as f:
        lines = f.readlines()
        lines2 = f2.readlines()
        for line, line2 in zip(lines, lines2):
            token_level_sum += len(line.strip().split(" "))
            arc_level_sum += len(line2.strip().split(","))
            extra_token_sum += len(line2.strip().split(",")) - len(line.strip().split(" "))
    
    print(f"  Token level sum: {token_level_sum}")
    print(f"  Arc level sum: {arc_level_sum}")
    print(f"  Extra tokens: {extra_token_sum}")
    print(f"  Extra tokens / Token level ratio: {round((extra_token_sum *100)/token_level_sum)}%")
    print(f"  Token per line: {token_level_sum/len(lines)}")
    print(f"  Arc token per line: {arc_level_sum/len(lines)}")
    print(f"  Extra tokens per line: {extra_token_sum/len(lines)}\n")
    f2.close()



if __name__ == "__main__":
    SPLIT = ["TRAIN", "TEST", "DEV"]
    SDP = ["dm", "pas", "psd"]
    for split in SPLIT:
        for sdp in SDP:
            print(f"{split} data distrubution:\n")
            arc_point_file = f"./arc_with_point/BLLIP_LG_{split}_{sdp}.csv"
            token_level_file = f"./token_level/BLLIP_LG_{split}_SPM_TOK.txt"
            compete_token_num(token_level_file, arc_point_file, sdp)