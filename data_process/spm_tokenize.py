import sentencepiece as spm


def train_spm_model():
    # train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
    # `m.vocab` is just a reference. not used in the segmentation.
    
    # spm.SentencePieceTrainer.train("""--input=BLLIP_LG_TRAIN.txt --model_prefix=BLLIP_spm --vocab_size=12000 --character_coverage=1
    # --eos_id=2 --unk_id=3 --pad_id=0 --bos_id=1 --shuffle_input_sentence=true --max_sentence_length=100000
    # --user_defined_symbols=->,<-,->2,<-2""")
    
    #DTG
    spm.SentencePieceTrainer.Train(input='BLLIP_LG_TRAIN.txt', model_prefix='./spm/BLLIP_spm',vocab_size=32719, \
                               character_coverage=1.0, pad_id=0, bos_id=1, eos_id=2, unk_id=3, \
                               user_defined_symbols='->,<-,->2,<-2', \
                               max_sentence_length=100000, shuffle_input_sentence=True
                               )

def spm_encode():
    sp = spm.SentencePieceProcessor()
    sp.load('./spm_size/BLLIP_spm.model')

    Type = ["TRAIN", "DEV", "TEST"]
    Dataset_type = "token_level_vocab"
    # SDP = ["dm", "pas", "psd"]
    SDP = ["TOK"]
    for t in Type:
        for sdp in SDP:
            print(f"Wirte file : ./{Dataset_type}/BLLIP_LG_{t}_SPM_ARC_{sdp}.txt")
            fw = open(f'./{Dataset_type}/BLLIP_LG_{t}_SPM_ARC_{sdp}.csv', 'w')
            # with open(f'./{Dataset_type}/BLLIP_LG_{t}_{sdp}.txt', 'r') as f:
            with open(f'./BLLIP_LG_{t}.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    encoder_list = sp.encode_as_ids(line)
                    fw.write(','.join(map(str, encoder_list))+'\n')
            f.close()
            fw.close()
            print("Done.")

if __name__ == '__main__':
    # train_spm_model()
    spm_encode()