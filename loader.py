# coding=utf-8
import numpy as np
import scipy.sparse as sp
import json
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from util.nlp_utils import *


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def bc_accuracy(output, labels):
    preds = output >= 0.5
    preds = preds.cpu().float().view(-1)
    correct = preds.eq(labels.cpu().float()).double()
    correct = correct.sum()
    return correct / len(labels)

def bc_accuracy2(output, labels):
    preds = output >= 0.5
    preds = preds.cpu().float().view(-1)
    #labels = labels.cpu().float().view(-1)
    labels = torch.stack(labels).cpu().float().view(-1)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_f1_threeClass(pred:torch.Tensor, labels:list)-> float:
    '''
    QWC
    Accuracy and f1 calculation of three classifications
    @Param  
        pred: tensor, the distribution of three categories, tensor.shape:(total_size,3);  
        labels:   list [tensor(0),tensor(1),tensor(2)...]
    @return accuracy and f1
    '''

    pred_class = torch.argmax(pred,dim=-1) # tensor.shape:(total_size,)
    if pred_class.device.type != 'cpu':
        pred_class = pred_class.cpu()
    pred_class = pred_class.float()


    labels = torch.stack(labels) # tensor([0,1,2,...])
    if labels.device.type != 'cpu':
        labels = labels.cpu()
    labels = labels.float()

    # print("pred_class = {}".format(pred_class))
    # print("labels = {}".format(labels))

    correct = pred_class.eq(labels).double()
    correct = correct.sum().item()
    acc = correct / len(labels)

    f1 = f1_score(labels.numpy(), pred_class.numpy(), average="macro")

    return acc, f1


def f1score(output, label):
    preds = output >= 0.5
    preds = preds.float().view(-1)
    result = f1_score(label.cpu().numpy(), preds.cpu().numpy(), pos_label=1, average="binary")
    return result

def f1score2(output, label):
    preds = output >= 0.5
    preds = preds.float().view(-1)
    label = torch.stack(label).cpu().float().view(-1)
    #print('label:{}'.format(label.cpu().numpy()))
    #print('pred:{}'.format(preds.cpu().numpy()))
    result = f1_score(label.cpu().numpy(), preds.cpu().numpy(), pos_label=1, average="binary")
    return result


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph_data(path, word_to_ix, max_len, num_data):
    """
    :param path: graph data file path
    :param word_to_ix: transform the word sentence the sequence
    :param max_len: maximum length of text extracted from one document in each vertice
    :return:
    """
    labels_list = []  # doc pair relationship labels

    g_features_list = []  # doc pair global feature vectors
    g_multi_scale_features_list = []  # doc pair multi-scale feature vectors
    g_all_features_list = []  # doc pair all feature vectors

    g_vertices_betweenness_list = []  # betweenness scores of vertices
    g_vertices_pagerank_list = []  # page rank scores of vertices
    g_vertices_katz_list = []  # katz scores of vertices

    v_features_list = []
    v_texts_w2v_idxs_l_list = []
    v_texts_w2v_idxs_r_list = []

    adjs_numsent_list = []
    adjs_tfidf_list = []
    adjs_position_list = []
    adjs_textrank_list = []

    num_samples = 0

    fin = open(path, "r", encoding="utf-8")
    for line in fin:  # each line is a json data that stores a CCIG/CIG graph
        g = json.loads(line)

        # labels
        labels_list.append(g["label"])

        # global doc pair features
        g_features_list.append(g["g_features_vec"])  # similarities
        g_multi_scale_features_list.append(g["g_multi_scale_features_vec"])  # multi-scale similarities
        g_all_features_list.append(g["g_all_features_vec"])  # multi-scale similarities + time gap + topic

        # vertices features
        v_features_list.append(torch.FloatTensor(g["v_features_mat"]))

        # vertices scores
        g_vertices_betweenness_list.append(g["g_vertices_betweenness_vec"])
        g_vertices_pagerank_list.append(g["g_vertices_pagerank_vec"])
        g_vertices_katz_list.append(g["g_vertices_katz_vec"])

        # TODO: handle non-w2v case, i.e., the features are just vertices' numerical features
        v_texts = g["v_texts_mat"]
        word_idxs = []
        for i, row in enumerate(v_texts):
            word_idxs.append([])
            for j, val in enumerate(row):
                val = val.encode("utf8").split()
                sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
                word_idxs[i].append(sent_idx)
        word_idxs_l = torch.LongTensor([s[0] for s in word_idxs])
        word_idxs_r = torch.LongTensor([s[1] for s in word_idxs])
        v_texts_w2v_idxs_l_list.append(word_idxs_l)
        v_texts_w2v_idxs_r_list.append(word_idxs_r)

        # adjacent matrices
        adj_numsent = g["adj_mat_numsent"]
        adj_numsent = sp.coo_matrix(adj_numsent, shape=(len(adj_numsent), len(adj_numsent)), dtype=np.float32)
        adj_numsent = normalize(adj_numsent)
        adj_numsent = sparse_mx_to_torch_sparse_tensor(adj_numsent)
        adjs_numsent_list.append(adj_numsent)

        adj_tfidf = g["adj_mat_tfidf"]
        adj_tfidf = sp.coo_matrix(adj_tfidf, shape=(len(adj_tfidf), len(adj_tfidf)), dtype=np.float32)
        adj_tfidf = normalize(adj_tfidf)
        adj_tfidf = sparse_mx_to_torch_sparse_tensor(adj_tfidf)
        adjs_tfidf_list.append(adj_tfidf)

        adj_position = g["adj_mat_position"]
        adj_position = sp.coo_matrix(adj_position, shape=(len(adj_position), len(adj_position)), dtype=np.float32)
        adj_position = normalize(adj_position)
        adj_position = sparse_mx_to_torch_sparse_tensor(adj_position)
        adjs_position_list.append(adj_position)

        adj_textrank = g["adj_mat_textrank"]
        adj_textrank = sp.coo_matrix(adj_textrank, shape=(len(adj_textrank), len(adj_textrank)), dtype=np.float32)
        adj_textrank = normalize(adj_textrank)
        adj_textrank = sparse_mx_to_torch_sparse_tensor(adj_textrank)
        adjs_textrank_list.append(adj_textrank)

        num_samples = num_samples + 1

        if num_samples >= num_data:
            break

    # labels
    labels = torch.FloatTensor(labels_list)

    # global doc-pair features
    g_features_list = np.array(g_features_list)
    g_features = [torch.FloatTensor(np.array(x)) for x in g_features_list]

    g_multi_scale_features_list = np.array(g_multi_scale_features_list)
    g_multi_scale_features = [torch.FloatTensor(np.array(x)) for x in g_multi_scale_features_list]

    g_all_features_list = np.array(g_all_features_list)
    g_topic_feature = g_all_features_list[:, -1:]  # NOTICE: see feature_extractor.py, the last one is topic feature
    enc = OneHotEncoder()
    g_topic_feature = enc.fit_transform(g_topic_feature).toarray()
    g_all_features_list = np.concatenate((g_all_features_list[:, :-1], g_topic_feature), axis=1).tolist()
    g_all_features = [torch.FloatTensor(np.array(x)) for x in g_all_features_list]

    # vertices scores
    g_vertices_betweenness = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_betweenness_list]
    g_vertices_pagerank = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_pagerank_list]
    g_vertices_katz = [torch.FloatTensor(np.array(x[:-1])) for x in g_vertices_katz_list]

    # split train:valid:test = 0.6:0.2:0.2
    train_bound = int(num_samples * 0.6)
    eval_bound = int(num_samples * (0.6 + 0.2))
    idx_train = torch.LongTensor(range(train_bound))
    idx_val = torch.LongTensor(range(train_bound, eval_bound))
    idx_test = torch.LongTensor(range(eval_bound, num_samples))

    return v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list,\
        adjs_numsent_list, adjs_tfidf_list, adjs_position_list, adjs_textrank_list,\
        g_features, g_multi_scale_features, g_all_features,\
        g_vertices_betweenness, g_vertices_pagerank, g_vertices_katz,\
        labels, idx_train, idx_val, idx_test


def load_one_graph(path, num_data):
    """
    :param path: graph data file path
    :param word_to_ix: transform the word sentence the sequence
    :param max_len: maximum length of text extracted from one document in each vertice
    :return:
    """
    labels_list = []  # doc pair relationship labels

    title1_list = []
    title2_list = []

    v_texts_w2v_idxs_l_list = []
    v_texts_w2v_idxs_r_list = []

    g1_concept_list = []
    g2_concept_list = []

    adjs_textrank_list = []

    num_samples = 0

    fin = open(path, "r", encoding="utf-8")
    #i = 1
    for line in fin:  # each line is a json data that stores a CCIG/CIG graph
        #print('{}-th row is WRONG!!!'.format(i))
        g = json.loads(line)

        # labels
        labels_list.append(int(g["label"]))

        # title
        # 对于BERT，不需要把word转成id
        title1 = g["sentence1_OOV"]
        #title1 = ''
        title1_list.append(title1)
        #word_idxs = []
        #for j, val in enumerate(title1):
        #    val = val.encode("utf8").split()
        #    sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #    word_idxs.append(sent_idx)
        #word_idxs_l = torch.LongTensor([s for s in word_idxs])
        #title1_list = word_idxs_l

        title2 = g["sentence2_OOV"]
        #title2 = ''
        title2_list.append(title2)

        #word_idxs = []
        #for j, val in enumerate(title2):
        #    val = val.encode("utf8").split()
        #    sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #    word_idxs.append(sent_idx)
        #word_idxs_l = torch.LongTensor([s for s in word_idxs])
        #title2_list = word_idxs_l

        # TODO: handle non-w2v case, i.e., the features are just vertices' numerical features
        #v_texts = g["TOP5_1"]  # g["TOP5_1"]
        v_texts = g["sentence1"]
        v_texts_w2v_idxs_l_list.append(v_texts)
        #word_idxs = []
        #for i, row in enumerate(v_texts):
        #    word_idxs.append([])
        #    for j, val in enumerate(row):
        #        val = val.encode("utf8").split()
        #       sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #        word_idxs[i].append(sent_idx)
        #word_idxs_l = torch.LongTensor([s[0] for s in word_idxs])
        #v_texts_w2v_idxs_l_list.append(word_idxs_l)
        v_texts = g["sentence2"]
        #v_texts = g["TOP5_2"]  # g["TOP5_2"]
        v_texts_w2v_idxs_r_list.append(v_texts)

        #word_idxs = []
        #for i, row in enumerate(v_texts):
        #    word_idxs.append([])
        #    for j, val in enumerate(row):
        #        val = val.encode("utf8").split()
        #        sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #        word_idxs[i].append(sent_idx)
        #word_idxs_r = torch.LongTensor([s[1] for s in word_idxs])
        #v_texts_w2v_idxs_r_list.append(word_idxs_r)
        #v_concept = g['concept1']
        v_concept = ''
        g1_concept_list.append(v_concept)

        #v_concept = g['concept2']
        v_concept = ''
        g2_concept_list.append(v_concept)

        if num_samples >= num_data:
            break

    # labels
    labels = torch.FloatTensor(labels_list)

    # no shuffle!!!
    # split train:valid:test = 0.6:0.2:0.2
    #train_bound = int(num_samples * 0.6)
    #eval_bound = int(num_samples * (0.6 + 0.2))
    #idx_train = torch.LongTensor(range(train_bound))
    #idx_val = torch.LongTensor(range(train_bound, eval_bound))
    #idx_test = torch.LongTensor(range(eval_bound, num_samples))

    # import pdb
    # pdb.set_trace()
    # print(labels)
    # print(title1_list)
    # print(v_texts_w2v_idxs_l_list)



    return labels, title1_list, title2_list, v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, g1_concept_list, g2_concept_list


def load_ELAM(path):
    """
    QWC edit
    :param path:  file path
    :return: [labels_list, text1_list, text2_list, relation_label_list,key_sentences_list]
    """
    labels_list = []  # doc pair relationship labels

    text1_list = []
    text2_list = []

    relation_label_list = []
    key_sentences_list = []

    fin = open(path, "r", encoding="utf-8")
    #i = 1
    for line in fin:  # each line is a json data that stores a CCIG/CIG graph
        g = json.loads(line)

        # labels
        labels_list.append(int(g["label"]))

        # text
        text1 = g["case_A"][0]  # list of str
        text1_list.append(text1)
        text2 = g["case_B"][0]  # list of str
        text2_list.append(text2)

        #relation_label
        relation_label = g["relation_label"]
        total_rl = relation_label["relation_label_aqss"] + relation_label["relation_label_yjss"] + relation_label["relation_label_zyjd"]#type:list
        total_rl = map(tuple,total_rl) #为了去重，变成固定的tuple
        total_rl = list(set(total_rl)) #去重.......[(0,0),(1,5)]
        relation_label_list.append(total_rl)

        # key sentences  
        ks1 = g["case_A"][-1]  #[[4, 1], [5, 1], [6, 1], [7, 1], [13, 2], [15, 2]]
        ks2 = g["case_B"][-1]
        key_sentences_list.append([ks1,ks2])
        

        

    # labels

    labels_list = torch.FloatTensor(labels_list)

    return [labels_list, text1_list, text2_list, relation_label_list, key_sentences_list]




def load_one_graph_2(path, word_to_ix, max_len, num_data):
    """
    :param path: graph data file path
    :param word_to_ix: transform the word sentence the sequence
    :param max_len: maximum length of text extracted from one document in each vertice
    :return:
    """
    labels_list = []  # doc pair relationship labels

    title1_list = []
    title2_list = []

    v_texts_w2v_idxs_l_list = []
    v_texts_w2v_idxs_r_list = []

    g1_concept_list = []
    g2_concept_list = []

    adjs_textrank_list = []

    num_samples = 0

    fin = open(path, "r", encoding="utf-8")
    #i = 1
    for line in fin:  # each line is a json data that stores a CCIG/CIG graph
        #print('{}-th row is WRONG!!!'.format(i))
        g = json.loads(line)

        # labels
        labels_list.append(int(g["label"]))

        # title
        # 对于BERT，不需要把word转成id
        title1 = g['sentence1_OOV']
        title1_list.append(title1)
        #word_idxs = []
        #for j, val in enumerate(title1):
        #    val = val.encode("utf8").split()
        #    sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #    word_idxs.append(sent_idx)
        #word_idxs_l = torch.LongTensor([s for s in word_idxs])
        #title1_list = word_idxs_l

        title2 = g['sentence2_OOV']
        title2_list.append(title2)

        #word_idxs = []
        #for j, val in enumerate(title2):
        #    val = val.encode("utf8").split()
        #    sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #    word_idxs.append(sent_idx)
        #word_idxs_l = torch.LongTensor([s for s in word_idxs])
        #title2_list = word_idxs_l

        # TODO: handle non-w2v case, i.e., the features are just vertices' numerical features
        #v_texts = g["TOP10_1"]  # g["TOP5_1"]
        v_texts = g["sentence1"]
        #v_texts = g["sentence1"]
        v_texts_w2v_idxs_l_list.append(v_texts)
        #word_idxs = []
        #for i, row in enumerate(v_texts):
        #    word_idxs.append([])
        #    for j, val in enumerate(row):
        #        val = val.encode("utf8").split()
        #       sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #        word_idxs[i].append(sent_idx)
        #word_idxs_l = torch.LongTensor([s[0] for s in word_idxs])
        #v_texts_w2v_idxs_l_list.append(word_idxs_l)
        v_texts = g["sentence2"]
        #v_texts = g["TOP10_2"]  # g["TOP5_2"]
        v_texts_w2v_idxs_r_list.append(v_texts)

        #word_idxs = []
        #for i, row in enumerate(v_texts):
        #    word_idxs.append([])
        #    for j, val in enumerate(row):
        #        val = val.encode("utf8").split()
        #        sent_idx = right_pad_zeros_1d([word_to_ix[w.decode("utf-8")] for w in val], max_len)
        #        word_idxs[i].append(sent_idx)
        #word_idxs_r = torch.LongTensor([s[1] for s in word_idxs])
        #v_texts_w2v_idxs_r_list.append(word_idxs_r)

        #v_concept = g['concept1']
        v_concept = ''
        g1_concept_list.append(v_concept)

        #v_concept = g['concept2']
        v_concept = ''
        g2_concept_list.append(v_concept)

        if num_samples >= num_data:
            break

    # labels
    labels = torch.FloatTensor(labels_list)

    # no shuffle!!!
    # split train:valid:test = 0.6:0.2:0.2
    #train_bound = int(num_samples * 0.6)
    #eval_bound = int(num_samples * (0.6 + 0.2))
    #idx_train = torch.LongTensor(range(train_bound))
    #idx_val = torch.LongTensor(range(train_bound, eval_bound))
    #idx_test = torch.LongTensor(range(eval_bound, num_samples))

    return labels, title1_list, title2_list, v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, g1_concept_list, g2_concept_list


def load_simcse(path):
    labels_list = []  # doc pair relationship labels

    text1_tensor_list, text2_tensor_list, weight1_tensor_list, weight2_tensor_list, cls_tensor_list, sep_tensor_list = \
        [], [], [], [], [], [],

    # fin = open(path, "r", encoding="utf-8")
    with open(path, "r") as fin:
        num = 1
        for line in fin:
            # print('path:{}\nnum:{}'.format(path, num))
            num += 1
            g = json.loads(line)

            # labels
            labels_list.append(int(g["label"]))
            text1_tensor_list.append(g["text1"])
            text2_tensor_list.append(g["text2"])
            weight1_tensor_list.append(g["weight1"])
            weight2_tensor_list.append(g["weight2"])
            cls_tensor_list.append(g["CLS"])
            sep_tensor_list.append(g["SEP"])

        # labels
        labels = torch.FloatTensor(labels_list)

    return labels, text1_tensor_list, text2_tensor_list, weight1_tensor_list, weight2_tensor_list, cls_tensor_list, \
           sep_tensor_list
