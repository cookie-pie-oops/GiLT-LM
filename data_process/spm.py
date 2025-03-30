import sentencepiece as spm

def spm_encode():
    sp = spm.SentencePieceProcessor()
    sp.load('./spm_parsing/BLLIP_spm.model')
    # encode with TG_spm and (65, 32762) == ->, (3, 65) == <-, (3, 2371) == <-2, (65, 32762, 553) == ->2
    # -> 32768 <- 32769 ->2 32770 <-2 32771

    # Type = ["TEST","DEV","TRAIN"]
    Type = ["TEST"]
    Dataset_type = "sdp_trans_silver_parse_graph"
    # SDP = ["dm", 'pas', 'psd']
    SDP = ["psd"]
    num = 1800
    # SDP = ["TOK"]
    for t in Type:
        for sdp in SDP:
            print(f"Wirte file : ./{Dataset_type}/BLLIP_LG_{t}_SPM_{sdp}_{num}.txt")
            fw = open(f'./{Dataset_type}/BLLIP_LG_{t}_SPM_{sdp}_{num}.txt', 'w')
            with open(f'./{Dataset_type}/BLLIP_LG_{t}_{sdp}_{num}.txt', 'r') as f:
            # with open(f'./BLLIP_LG_{t}.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    encoder_list = sp.encode_as_ids(line)
                    fw.write(' '.join(map(str, encoder_list))+'\n')
            f.close()
            fw.close()
            print("Done.")