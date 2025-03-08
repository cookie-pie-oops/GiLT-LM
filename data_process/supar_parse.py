from supar import Parser
import hanlp
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import json
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  

def tag_to_wordnet_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_lemma(batched_line, tagger):
    batched_tags = tagger(batched_line)
    output_triple = []
    for line, tags in zip(batched_line, batched_tags):
        line_triple = []
        for (word, tag) in zip(line, tags):
            if tag_to_wordnet_tag(tag):
                lemma_word = wnl.lemmatize(word, tag_to_wordnet_tag(tag))
            else:
                lemma_word = wnl.lemmatize(word)
            triple = (word, lemma_word, tag)
            line_triple.append(triple)
        output_triple.append(line_triple)
    return output_triple


def write_into_file(dataset, dataset_name, sdp_name):
    f = open(f"./parser/{dataset_name}_Supar_test.json", "a")
    for ds in dataset:
        word_ids, words, lemmas, tags, _, _, _, _, heads, _ = ds.values
        json_obj = {"text": " ".join(words)}
        sdp_graph = []
        for head, word, word_id, lemma, tag in zip(heads, words, word_ids, lemmas, tags):
            head_list = []
            label_list = []
            if head != "_":
                if "|" in head:
                    head = head.split("|")
                else:
                    head = [head]
                for item in head:
                    head_id = int(item.split(":")[0])
                    arc_label = item.split(":")[1]
                    head_list.append(head_id)
                    label_list.append(arc_label)
            sdp_graph.append({
                "id": word_id,
                "form": word,
                "head": head_list,
                "label": label_list,
                "cpos": tag,
                "lemma": lemma
                })
        json_obj[f"{sdp_name}_graph"] = sdp_graph
        f.write(json.dumps(json_obj) + "\n")
    f.close()


def read_sdp_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            if line:
                parts = line.split('\t')
                if len(parts) > 1:
                    sentence.append(parts)
        if sentence:
            sentences.append(sentence)
    return sentences


def write_sdp_file(file_path, sentences):
    with open(file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write('# \n')
            for token in sentence:
                file.write('\t'.join(token) + '\n')
            file.write('\n')


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_unlabeled_f1(gt_label, pred_ds):
    sent_infer = 0
    sent_correct = 0
    sent_label = 0
    for ds, gt_sent in zip(pred_ds, gt_label):
        word_ids, words, _, _, _, _, _, _, heads, _ = ds.values

        for head, gt_word in zip(heads, gt_sent):         
            count_infer = {}
            count_label = {}
            for i in gt_word:
                count_label[i] = count_label.get(i, 0) + 1
                sent_label += 1

            if head != "_":
                if "|" in head:
                    head = head.split("|")
                else:
                    head = [head]
                for item in head:
                    arc_label = item.split(":")[1]
                    if arc_label != "root":
                        count_infer[arc_label] = count_infer.get(arc_label, 0) + 1
                        sent_infer += 1
            
            for key, value in count_infer.items():
                if key in count_label:
                    sent_correct += min(value, count_label[key])
    
    pre = sent_correct / sent_infer if sent_infer > 0 else 0
    recall = sent_correct / sent_label if sent_label > 0 else 0
    f1 = 2 * pre * recall / (pre + recall) if pre + recall > 0 else 0
    return pre, recall, f1


def evaluate_biaffine_and_MFVI():
    sdp_path = "/home/huangty/SDP_Transformer_project/Dataset/Semeval-2014/2014/test/dm.sdp"
    df = read_sdp_file(sdp_path)
    sentences = []
    sentences_label = []
    for df_sent in df:
        sentence = []
        sent_label = []
        for df_word in df_sent:
            word_label = []
            sentence.append(df_word[1])
            for i in df_word[6:]:
                if i != "_":
                    word_label.append(i)
            sent_label.append(word_label)
        sentences.append(sentence)
        sentences_label.append(sent_label)
    
    # parser = Parser.load('biaffine-sdp-en')
    parser = Parser.load('vi-sdp-roberta-en')
    print("load success!")
    tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    wnl = WordNetLemmatizer()

    dataset_wolemma = parser.predict(sentences, prob=True)

    output_triple = get_lemma(sentences, tagger)
    dataset_withlemma = parser.predict(output_triple, prob=True)
    # for i in range(len(dataset_withlemma)):
    #     if dataset_withlemma.labels[i] != dataset_wolemma.labels[i]:
    #         print(i)

    pre, recall, f1 = get_unlabeled_f1(sentences_label, dataset_withlemma)
    print(f"with lemma: pre: {pre}, recall: {recall}, f1: {f1}")
    pre, recall, f1 = get_unlabeled_f1(sentences_label, dataset_wolemma)
    print(f"w/o lemma: pre: {pre}, recall: {recall}, f1: {f1}")


if __name__ == '__main__':
    # nltk.download('wordnet')

    # parser = Parser.load('biaffine-sdp-en')
    parser = Parser.load('vi-sdp-roberta-en')
    print("load success!")
    tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
    wnl = WordNetLemmatizer()

    # datasets = ["DEV", "TEST", "TRAIN"]
    # sdp_name = "dm"
    # for dataset_name in datasets:
    #     txt_file = open(f"./BLLIP_LG_{dataset_name}.txt", "r")
    #     lines = txt_file.readlines()
    #     batch_size = 256
    #     sents = [line.strip().split(" ") for line in lines]
    #     batched_lines = [sents[i:i+batch_size] for i in range(0, len(sents), batch_size)]

    #     for batched_line in batched_lines:
    #         output_triple = get_lemma(batched_line, tagger)
    #         dataset = parser.predict(output_triple, prob=True)
    #         # prob = dataset.probs
    #         import pdb;pdb.set_trace()
    #         write_into_file(dataset, dataset_name, sdp_name)