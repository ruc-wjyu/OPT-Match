# coding=utf-8
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import ot
from .sinkhorn import *
from .utils import *
from transformers import BertModel, BertConfig,BertTokenizer
import Levenshtein
from torch.nn.parameter import Parameter
from torch.nn import init


torch.set_printoptions(profile="full")

class Matching_Component(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/home/weijie_yu/pretrain_models/bert-base-uncased/vocab.txt')
        self.modelConfig = BertConfig.from_pretrained('/home/weijie_yu/pretrain_models/bert-base-uncased/config.json')
        self.bert_model = BertModel.from_pretrained('/home/weijie_yu/pretrain_models/bert-base-uncased/pytorch_model.bin', config=self.modelConfig)

    def forward(self, text1, text2):

        txt1_token = self.tokenizer.encode(text1[:512], add_special_tokens=False)
        txt2_token = self.tokenizer.encode(text2[:512], add_special_tokens=False)
        indexed_tokens_1 = [self.tokenizer.convert_tokens_to_ids("[CLS]")] + txt1_token
        if len(indexed_tokens_1) > 256:
            indexed_tokens_1 = indexed_tokens_1[:256]
        segments_ids_1 = [0] * len(indexed_tokens_1)

        indexed_tokens_2 =  txt2_token + [self.tokenizer.convert_tokens_to_ids("[SEP]")]
        if len(indexed_tokens_2) > 256:
            indexed_tokens_2 = indexed_tokens_2[:256]

        segments_ids_2 = [1] * len(indexed_tokens_2)

        indexed_tokens = indexed_tokens_1 + indexed_tokens_2
        segments_ids = segments_ids_1 + segments_ids_2

        segments_tensors = torch.tensor([segments_ids]).cuda()
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()

        #with torch.no_grad():
        encoded_layers, _ = self.bert_model(tokens_tensor, token_type_ids=segments_tensors)
        cls_rep = encoded_layers.squeeze(0)[0]

        return cls_rep


class SelectSentence(nn.Module):
    def __init__(self, args, hidden_dim, dropout):
        super(SelectSentence, self).__init__()

        # network configure
        self.args = args
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.bert_pair = Matching_Component()
        self.bert_dim = 768

        self.regressor = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bert_dim, self.hidden_dim), # self.hidden_dim * 4 or 8, 单个sentence或者concept是一个full_fusion就是4，拼接就是8
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid())

        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(self.bert_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.cls.weight, -initrange, initrange)

    def reset_para(self):
        init.kaiming_uniform_(self.weight_matrix, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_para, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_para)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.att_para, -bound, bound)


    def plot(self, sentence_rep_1, sentence_rep_2, epoch, metric='cosine'):
        D = self.cost_matrix(sentence_rep_1, sentence_rep_2, metric)
        path = './pic/no_title_'+ str(epoch) + '_event_top5.pdf'
        plt.imshow(D, cmap='Purples', vmin=0.9, vmax=1)  # norm=norm,
        plt.show()
        plt.savefig(path)

    def save_file(self, sentence_rep_1, sentence_rep_2, epoch, metric='cosine'):
        D = self.cost_matrix(sentence_rep_1, sentence_rep_2)
        write_path = './vector/title_'+ str(epoch) + '_event_top5.txt'
        np.savetxt(write_path, D)


    def gumbel_softmax(self, logits, temperature=0.5, hard=False):
        """
        for soft selection
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard


    def forward(self, g1_text, g2_text, g1_title_text, g2_title_text, g1_concept, g2_concept, batch_size, reg, iter, sigma, epoch, args
                ):
        # for hard selection
        prediction = []
        for i in range(batch_size):
            title1 = g1_text[i]
            title2 = g2_text[i]

            text1 = g1_text[i]
            text2 = g2_text[i]

            len1, len2 = len(text1), len(text2)
            D = torch.empty([len1, len2])
            for i in range(len1):
                for j in range(len2):
                    weight = my_common_words(text1[i],text2[j])
                    #weight = my_common_words2(text1[i],text2[j])
                    #weight = my_common_keywords2(text1[i],text2[j], concept1, concept2)
                    #weight = Levenshtein_ratio(text1[i],text2[j])
                    #weight = Levenshtein_jaro_winkler(text1[i],text2[j])
                    if weight == 0.:
                        D[i][j] = 99.
                    else:
                        D[i][j] = -weight*50.0

            n, m = len1, len2

            margin_1 = torch.ones(n)/n
            margin_1 = margin_1.cuda()
            margin_2 = torch.ones(m) / m
            margin_2 = margin_2.cuda()
            mass = torch.min(torch.sum(margin_1),torch.sum(margin_2))/sigma


            if args.OT:
                T = sinkhorn_knopp(margin_1, margin_2, D, reg, iter) # M
            else:
                T = entropic_partial_wasserstein(margin_1, margin_2, D, reg, mass, iter)  # M

            #write_path = '/home/weijie_yu/dataset/MDA_data/new/S2ORC/plot/edit_OPT_sigma4.txt'
            #plan = T.cpu()
            #np.savetxt(write_path, plan)

            txt1 = []
            txt2 = []
            m1 = T.sum(dim=-1)
            m2 = T.sum(dim=0)
            topk =  T.shape[0] if T.shape[0]<=T.shape[1] else T.shape[1]
            #topk = 5
            if topk < 5:
                _, top_index_1 = torch.topk(m1, topk)
                _, top_index_2 = torch.topk(m2, topk)
            else:
                _, top_index_1 = torch.topk(m1, 5)
                _, top_index_2 = torch.topk(m2, 5)
            index_1 = top_index_1.cpu().numpy()
            index_2 = top_index_2.cpu().numpy()

            for k in index_1:
                txt1.append(title1[k]) # [:100] for pla

            for k in index_2:
                txt2.append(title2[k])

            text1 = ' '.join(txt1)
            text2 = ' '.join(txt2)

            cls = self.bert_pair(text1, text2)
            pred = self.cls(cls)

            prediction.append(self.sigmoid(pred))


        return prediction

