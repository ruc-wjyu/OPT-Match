# coding=utf-8
import time
import argparse
import math
import torch.nn as nn
import torch.optim as optim
from loader import *
from models import *
from models.ema import EMA
from util.nlp_utils import *
from util.exp_utils import set_device, set_random_seed, summarize_model
from tqdm import tqdm
import random
import torch.nn.functional as f
from transformers import AdamW
from models.model import SelectSentence

# Training settings
parser = argparse.ArgumentParser()

# cuda, seed
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# data files
parser.add_argument('--inputdata', type=str, default="event-story-cluster/same_story_doc_pair.cd.json", help='input data path')
parser.add_argument('--outputresult', type=str, default="event_result.txt", help='output file path')
parser.add_argument('--data_type', type=str, default="event", help='event or story')

# train
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--num_data', type=int, default=1000000000, help='maximum number of data samples to use.')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--lr_warm_up_num', default=500, type=int, help='number of warm-up steps of learning rate')
parser.add_argument('--beta1', default=0.8, type=float, help='beta 1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')
parser.add_argument('--no_grad_clip', default=False, action='store_true', help='whether use gradient clip')
parser.add_argument('--max_grad_norm', default=5.0, type=float, help='global Norm gradient clipping rate')
parser.add_argument('--use_ema', default=True, action='store_true', help='whether use exponential moving average')
parser.add_argument('--ema_decay', default=0.9999, type=float, help='exponential moving average decay')

# model
parser.add_argument('--hidden_vfeat', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout_vfeat', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_siamese', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout_siamese', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_final', type=int, default=16, help='Number of hidden units.')

parser.add_argument('--use_gcn', action='store_true', default=False, help='use GCN in model.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--use_cd', action='store_true', default=False, help='use community detection in graph.')
parser.add_argument('--use_siamese', action='store_true', default=False, help='use siamese encoding for vertices.')
parser.add_argument('--use_vfeatures', action='store_true', default=False, help='use vertex features.')
parser.add_argument('--use_gfeatures', action='store_true', default=False, help='use global graph features.')

parser.add_argument('--gfeatures_type', type=str, default="features", help='what features to use: features, multi_scale_features, all_features')
parser.add_argument('--gcn_type', type=str, default="valina", help='gcn layer type')
parser.add_argument('--pool_type', type=str, default="mean", help='pooling layer type')
parser.add_argument('--combine_type', type=str, default="separate", help='separate or concatenate, vfeatures with siamese encoding')

# graph
parser.add_argument('--adjacent', type=str, default="tfidf", help='adjacent matrix')
parser.add_argument('--vertice', type=str, default="pagerank", help='vertex centrality')
parser.add_argument('--betweenness_threshold_coef', type=float, default=1.0, help='community detection parameter')
parser.add_argument('--max_c_size', type=int, default=6, help='community detection parameter')
parser.add_argument('--min_c_size', type=int, default=2, help='community detection parameter')

# ywj edit
parser.add_argument('--OT', type=bool, default=False, help='OT or OPT')
parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
parser.add_argument('--balance', type=float, default=0.001, help='trade-off between matching and OT')
parser.add_argument('--reg', type=float, default=0.5, help='weight of entropy reg for Sinkhorn')
parser.add_argument('--iter', type=int, default=10, help='num of iteration of Sinkhorn')
parser.add_argument('--sigma', type=float, default=2.0, help='variance of diagonal gaussian prior')


args = parser.parse_args()

# configuration
device, use_cuda, n_gpu = set_device(args.no_cuda)
set_random_seed(args.seed)

if args.data_type == "AAN":
    args.train = "AAN/train_new.json"
    args.dev = "AAN/dev_new.json"
    args.test = "AAN/test_new.json"
    args.outputresult = "AAN_result.txt"

if args.data_type == "OC":
    args.train = "OC/train_new.json"
    args.dev = "OC/dev_new.json"
    args.test = "OC/test_new.json"
    args.outputresult = "OC_result.txt"

if args.data_type == "S2ORC":
    args.train = "S2ORC/train_new.json"
    args.dev = "S2ORC/dev_new.json"
    args.test = "S2ORC/test_new.json"
    args.outputresult = "S2ORC_result.txt"

if args.data_type == "pla":
    args.train = "pla/train_new.json"
    args.dev = "pla/dev_new.json"
    args.test = "pla/test_new.json"
    args.outputresult = "pla_result.txt"

print(args)

path = "/home/weijie_yu/dataset/MDA_data/new/"
train_path = path + args.train
dev_path = path + args.dev
test_path = path + args.test
print("begin loading DATA............" + path)


# test data
test_labels, test_title1_list, test_title2_list, test_text1_list, test_text2_list, test_concept1_list, test_concept2_list \
    = load_one_graph(test_path, args.num_data)


model = SelectSentence(args, 200, 0.2)
model = model.cuda()
summarize_model(model)

if args.use_ema:
    ema = EMA(args.ema_decay)
    ema.register(model)

# optimizer and scheduler
parameters = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = optim.Adam(
#    params=parameters,
#    lr=args.lr,
#    betas=(args.beta1, args.beta2),
#    eps=1e-8,
#    weight_decay=3e-7)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)  # lr=3e-6

'''
scheduler = None
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
'''
cr = 1.0 / math.log(args.lr_warm_up_num)
scheduler = None
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda ee: cr * math.log(ee + 1)
    if ee < args.lr_warm_up_num else 1)


def make_batch(labels, g1_title_list, g2_title_list, g1_text_list, g2_text_list, g1_concept_list, g2_concept_list,batch_size):

    num_batches_per_epoch = int((len(labels))/batch_size)

    batch_label = []
    batch_g1_title_list = []
    batch_g2_title_list = []
    batch_g1_text_list = []
    batch_g2_text_list = []
    batch_g1_concept_list = []
    batch_g2_concept_list = []
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        in_label = labels[start_index:end_index]
        in_g1_title_list = g1_title_list[start_index:end_index]
        in_g2_title_list = g2_title_list[start_index:end_index]
        in_g1_text_list = g1_text_list[start_index:end_index]
        in_g2_text_list = g2_text_list[start_index:end_index]
        in_g1_concept_list = g1_concept_list[start_index:end_index]
        in_g2_concept_list = g2_concept_list[start_index:end_index]

        batch_label.append(in_label)
        batch_g1_title_list.append(in_g1_title_list)
        batch_g2_title_list.append(in_g2_title_list)
        batch_g1_text_list.append(in_g1_text_list)
        batch_g2_text_list.append(in_g2_text_list)
        batch_g1_concept_list.append(in_g1_concept_list)
        batch_g2_concept_list.append(in_g2_concept_list)

    return batch_label, batch_g1_title_list, batch_g2_title_list, batch_g1_text_list, batch_g2_text_list, \
           batch_g1_concept_list, batch_g2_concept_list

def train_epoch(epoch, step):
    # train data
    labels, g1_title_list, g2_title_list, g1_text_list, g2_text_list, g1_concept_list, g2_concept_list \
        = load_one_graph(train_path, args.num_data)

    # dev data
    dev_labels, dev_g1_title_list, dev_g2_title_list, dev_g1_text_list, dev_g2_text_list, dev_g1_concept_list, dev_g2_concept_list \
        = load_one_graph(dev_path, args.num_data)

    t = time.time()
    outputs = []
    loss_train = 0.0

    # shuffle training data

    train_data = list(
        zip(labels, g1_title_list, g2_title_list, g1_text_list, g2_text_list, g1_concept_list, g2_concept_list))
    random.shuffle(train_data)
    labels, g1_title_list, g2_title_list, g1_text_list, g2_text_list, g1_concept_list, g2_concept_list = map(list, zip(
        *train_data))

    train_labels, train_g1_title_list, train_g2_title_list, train_g1_text_list, train_g2_text_list, train_g1_concept_list, train_g2_concept_list \
        = make_batch(labels, g1_title_list, g2_title_list, g1_text_list, g2_text_list, g1_concept_list, g2_concept_list,
                     args.batch_size)

    idx_train = len(train_labels)  # num batches
    labels = labels[:idx_train * args.batch_size]

    for i in range(idx_train):
        labels_batch = [x.cuda() for x in train_labels[i]]
        # calculate loss and back propagation
        model.train()
        optimizer.zero_grad()
        output = model(train_g1_text_list[i], train_g2_text_list[i], train_g1_title_list[i], train_g2_title_list[i],
                       train_g1_concept_list[i], train_g2_concept_list[i], args.batch_size, args.reg, args.iter,
                       args.sigma, epoch, args)

        for k in range(len(output)):
            outputs.append(output[k].data)

        output = torch.cat(output,dim=0)
        labels_batch = [x.unsqueeze(0) for x in labels_batch]
        labels_batch = torch.cat(labels_batch,dim=0)
        loss = nn.BCELoss()(output, labels_batch)

        loss_train += loss.item()
        loss.backward()

        # gradient clip
        if (not args.no_grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # update model
        optimizer.step()

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # exponential moving avarage
        if args.use_ema:
            ema(model, step)

        if i % int(idx_train/3) == 0:  #

            t = time.time()
            test_loss, test_acc, test_f1 = test(model, dev_labels, dev_g1_title_list, dev_g2_title_list, dev_g1_text_list,
                                     dev_g2_text_list, dev_g1_concept_list, dev_g2_concept_list, args.reg, args.iter, args.sigma, epoch, args.OT)

            print(
                "Test set results:",
                "loss= {:.4f}".format(test_loss),
                "accuracy= {:.4f}".format(test_acc),
                "f1= {:.4f}".format(test_f1),
                'time: {:.4f}s'.format(time.time() - t))

            fout.write(
                "Test set results:" +
                "loss= " + str(test_loss) + ',' +
                "accuracy= " + str(test_acc) + ',' +
                "f1_score= " + str(test_f1) + '\n')

        step += 1

    loss_train = loss_train / idx_train
    acc_train = bc_accuracy2(torch.stack(outputs), labels)
    f1_train = f1score2(torch.stack(outputs), labels)

    # Evaluate validation set performance separately,
    loss_val, acc_val, f1_val = test(model, dev_labels, dev_g1_title_list, dev_g2_title_list, dev_g1_text_list,
                                     dev_g2_text_list, dev_g1_concept_list, dev_g2_concept_list, args.reg, args.iter, args.sigma, epoch, args.OT)

    # print info
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'f1_train: {:.4f}'.format(f1_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'f1_val: {:.4f}'.format(f1_val),
          'time: {:.4f}s'.format(time.time() - t))
    fout.write('Epoch: ' + str(epoch + 1) + ',' +
               'loss_train: ' + str(loss_train) + ',' +
               'acc_train: ' + str(acc_train) + ',' +
               'f1_train: ' + str(f1_train) + ',' +
               'loss_val: ' + str(loss_val) + ',' +
               "acc_val: " + str(acc_val) + ',' +
               'f1_val: ' + str(f1_val) + '\n')
    return step


def train(args, fout):
    step = 0
    for epoch in range(args.epochs):
        step = train_epoch(epoch, step)
        t = time.time()
        test_loss, test_acc, test_f1 = test(model, test_labels, test_title1_list, test_title2_list,
                                            test_text1_list, test_text2_list, test_concept1_list, test_concept2_list,
                                            args.reg, args.iter, args.sigma, epoch, args.OT)
        print(
            "Test set results:",
            "loss= {:.4f}".format(test_loss),
            "accuracy= {:.4f}".format(test_acc),
            "f1= {:.4f}".format(test_f1),
            'time: {:.4f}s'.format(time.time() - t))

        fout.write(
            "Test set results:" +
            "loss= " + str(test_loss) + ',' +
            "accuracy= " + str(test_acc) + ',' +
            "f1_score= " + str(test_f1) + '\n')


def test(model, test_labels, test_g1_title_list, test_g2_title_list, test_g1_text_list, test_g2_text_list,
         test_concept1_list, test_concept2_list, reg, iter, sigma, epoch, mode):
    model.eval()
    outputs = []
    loss_test = 0.0

    dev_labels_batches, dev_g1_title_list_batches, dev_g2_title_list_batches, dev_g1_text_list_batches, \
    dev_g2_text_list_batches, dev_g1_concept_list_batches, dev_g2_concept_list_batches = \
        make_batch(test_labels, test_g1_title_list, test_g2_title_list, test_g1_text_list, test_g2_text_list,
                   test_concept1_list, test_concept2_list, args.batch_size)

    idxs = len(dev_labels_batches)  # num batches
    test_labels = test_labels[:idxs * args.batch_size]
    with torch.no_grad():
        for i in range(idxs):
            labels_batch = [x.cuda() for x in dev_labels_batches[i]]

            output = model(dev_g1_text_list_batches[i], dev_g2_text_list_batches[i], dev_g1_title_list_batches[i],
                           dev_g2_title_list_batches[i], dev_g1_concept_list_batches[i], dev_g2_concept_list_batches[i],
                           args.batch_size, reg, iter, sigma, epoch, args)


            for k in range(len(output)):
                outputs.append(output[k].data.cpu())

            output = torch.cat(output, dim=0)
            labels_batch = [x.unsqueeze(0) for x in labels_batch]
            labels_batch = torch.cat(labels_batch, dim=0)
            loss = nn.BCELoss()(output, labels_batch)
            loss_test += loss.data.item()


    loss = loss_test / idxs
    test_labels = [x.unsqueeze(0) for x in test_labels]
    acc = bc_accuracy2(torch.stack(outputs), test_labels)
    f1 = f1score2(torch.stack(outputs), test_labels)

    return loss, acc, f1


def write_to_file(fin, label_list, pred_list):
    with open(fin, 'w') as f:
        for i in range(len(label_list)):
            f.write(str(label_list[i]) + '\t' + str(pred_list[i]) + '\n')

def get_loss(logits, target):
    return f.cross_entropy(logits, target)

if __name__ == '__main__':
    t_total = time.time()
    outpath = "./" + args.outputresult
    print('Output path:{}'.format(outpath))
    fout = open(outpath, 'w')
    train(args, fout)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    fout.close()
