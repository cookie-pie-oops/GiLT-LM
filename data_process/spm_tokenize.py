import sentencepiece as spm
import re

def train_spm_model():
    # train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
    # `m.vocab` is just a reference. not used in the segmentation.
    
    # spm.SentencePieceTrainer.train("""--input=BLLIP_LG_TRAIN.txt --model_prefix=BLLIP_spm --vocab_size=12000 --character_coverage=1
    # --eos_id=2 --unk_id=3 --pad_id=0 --bos_id=1 --shuffle_input_sentence=true --max_sentence_length=100000
    # --user_defined_symbols=->,<-,->2,<-2""")
    
    #DTG
    spm.SentencePieceTrainer.Train(input='bllip_train_LG.cc', model_prefix='./spm/BLLIP_spm',vocab_size=32772, \
                               character_coverage=1.0, pad_id=0, bos_id=1, eos_id=2, unk_id=3, \
                               user_defined_symbols='->,<-,->2,<-2,(ADJP,(ADVP,(CONJP,(FRAG,(INTJ,(LST,(NAC,(NP,(NX,(PP,(PRN,(PRT,(QP,(RRC,(S,(SBAR,(SBARQ,(SINV,(SQ,(UCP,(VP,(WHADJP,(WHADVP,(WHNP,(WHPP,(X,ADJP),ADVP),CONJP),FRAG),INTJ),LST),NAC),NP),NX),PP),PRN),PRT),QP),RRC),S),SBAR),SBARQ),SINV),SQ),UCP),VP),WHADJP),WHADVP),WHNP),WHPP),X)', \
                               max_sentence_length=100000, shuffle_input_sentence=True
                               )

def spm_encode():
    sp = spm.SentencePieceProcessor()
    sp.load('./spm_parsing/BLLIP_spm.model')
    # encode with TG_spm and (65, 32762) == ->, (3, 65) == <-, (3, 2371) == <-2, (65, 32762, 553) == ->2
    # -> 32768 <- 32769 ->2 32770 <-2 32771

    Type = ["TEST","DEV","TRAIN"]
    # Type = ["TEST"]
    Dataset_type = "MRPC"
    # SDP = ["dm", 'pas', 'psd']
    SDP = ["psd"]
    # num = 1800
    # SDP = ["TOK"]
    for t in Type:
        for sdp in SDP:
            print(f"Wirte file : ./{Dataset_type}/{Dataset_type}_{t}_token.txt")
            fw = open(f'./{Dataset_type}/{Dataset_type}_{t}_token.txt', 'w')
            with open(f'./{Dataset_type}/{Dataset_type}_{t}.txt', 'r') as f:
            # with open(f'./BLLIP_LG_{t}.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    clean_line = re.sub(r'(?<!\s)[^\x00-\x7F](?!\s)', '', line)
                    encoder_list = sp.encode_as_ids(clean_line)
                    fw.write(','.join(map(str, encoder_list))+'\n')
            f.close()
            fw.close()
            print("Done.")

if __name__ == '__main__':
    # train_spm_model()
    spm_encode()