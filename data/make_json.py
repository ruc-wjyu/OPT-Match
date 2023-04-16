import csv
import json
from nltk.tokenize.punkt import PunktSentenceTokenizer
import re
import os
import pickle
import gensim
import numpy as np


TRAIN = '../../AAN/train_cite.csv'
DEV = '../../AAN/dev_cite.csv'
TEST = '.../../AAN/test_cite.csv'

OUT_TRAIN = './json/train_new.json'
OUT_DEV = './json/dev_new.json'
OUT_TEST = './json/test_new.json'


def load_w2v(fin, type, vector_size):
    """
    Load word vector file.
    :param fin: input word vector file name.
    :param type: word vector type, "Google" or "Glove" or "Company".
    :param vector_size: vector length.
    :return: Output Gensim word2vector model.
    """
    model = {}
    if type == "Google" or type == "Glove":
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fin, binary=True)
    elif type == "Company":
        model["PADDING"] = np.zeros(vector_size)
        model["UNKNOWN"] = np.random.uniform(-0.25, 0.25, vector_size)
        with open(fin, "r", encoding="utf-8") as fread:
            for line in fread.readlines():
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(" ".join(line_list[1:]),
                                         dtype=float, sep=" ")
                model[word] = word_vec
    else:
        print("type must be Glove or Google or Company.")
        sys.exit(1)
    print(type)
    return model


def split_english_sentence(text):
    """
    Segment a input English text into a list of sentences.
    :param text: a segmented input string.
    :return: a list of segmented sentences.
    """
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    sentences = [clean_text(i) for i in sentences]
    return sentences


def clean_text(text):
    text = text.lower().replace('- ', '').replace('\n', '.').replace('\t', '').replace('\r', '')
    cleaner = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    return cleaner.sub(' ', text)


def load_W2V_VOCAB():
    print("load w2v vocabulary ...")
    W2V_VOCAB_PKL_FILE = "/home/weijie_yu/Article_OT_EN/data/raw/Google-w2v/GoogleNews-vectors-negative300.vocab.pkl"
    if not os.path.exists(W2V_VOCAB_PKL_FILE):
        W2V = load_w2v("/home/weijie_yu/Article_OT_EN/data/raw/Google-w2v/GoogleNews-vectors-negative300.bin",
                       "Google", 300)
        W2V_VOCAB = set(W2V.wv.vocab.keys())  # must be a set to accelerate remove_OOV
        pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "wb"))
    else:
        W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "rb"))
    return W2V_VOCAB


def remove_OOV(text, vocab):
    """
    Remove OOV words in a text.
    """
    tokens = str(text).split()
    tokens = [word for word in tokens if word in vocab]
    new_text = " ".join(tokens)
    return new_text


W2V_VOCAB = load_W2V_VOCAB()

with open(OUT_DEV, 'w') as w1:
    with open(DEV, 'r') as r1:
        num = 1
        csv_reader = csv.reader(r1)
        for line in csv_reader:
            print('TEST---{}---###'.format(num))
            dict_g = {}
            dict_g["label"] = line[0]

            text1 = line[1]
            text1.replace("\001", "")
            sentences1 = split_english_sentence(text1)
            dict_g["sentence1_OOV"] = sentences1
            dict_g["sentence1"] = [remove_OOV(i, W2V_VOCAB) for i in sentences1]

            text2 = line[2]
            text2.replace("\001", "")
            sentences2 = split_english_sentence(text2)
            dict_g["sentence2_OOV"] = sentences2
            dict_g["sentence2"] = [remove_OOV(i, W2V_VOCAB) for i in sentences2]

            json_out = json.dumps(dict_g, ensure_ascii=False)  # to dump Chinese
            num += 1
            w1.write(json_out)
            w1.write('\n')

with open(OUT_TEST, 'w') as w2:
    with open(TEST, 'r') as r2:
        num = 1
        csv_reader = csv.reader(r2)
        for line in csv_reader:
            print('DEV---{}---###'.format(num))
            dict_g = {}
            dict_g["label"] = line[0]

            text1 = line[1]
            text1.replace("\001", "")
            sentences1 = split_english_sentence(text1)
            dict_g["sentence1_OOV"] = sentences1
            dict_g["sentence1"] = [remove_OOV(i, W2V_VOCAB) for i in sentences1]

            text2 = line[2]
            text2.replace("\001", "")
            sentences2 = split_english_sentence(text2)
            dict_g["sentence2_OOV"] = sentences2
            dict_g["sentence2"] = [remove_OOV(i, W2V_VOCAB) for i in sentences2]

            json_out = json.dumps(dict_g, ensure_ascii=False)  # to dump Chinese
            num += 1
            w2.write(json_out)
            w2.write('\n')

with open(OUT_TRAIN, 'w') as w3:
    with open(TRAIN, 'r') as r3:
        num = 1
        csv_reader = csv.reader(r3)
        for line in csv_reader:
            print('TRAIN---{}---###'.format(num))
            dict_g = {}
            dict_g["label"] = line[0]

            text1 = line[1]
            text1.replace("\001", "")
            sentences1 = split_english_sentence(text1)
            dict_g["sentence1_OOV"] = sentences1
            dict_g["sentence1"] = [remove_OOV(i, W2V_VOCAB) for i in sentences1]

            text2 = line[2]
            text2.replace("\001", "")
            sentences2 = split_english_sentence(text2)
            dict_g["sentence2_OOV"] = sentences2
            dict_g["sentence2"] = [remove_OOV(i, W2V_VOCAB) for i in sentences2]

            json_out = json.dumps(dict_g, ensure_ascii=False)  # to dump Chinese
            num += 1
            w3.write(json_out)
            w3.write('\n')