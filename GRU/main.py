# -*- coding: utf-8 -*-
from jieba import lcut
import json
from tqdm import tqdm
from torchtext.vocab import vocab
from collections import OrderedDict, Counter
from torchtext.transforms import VocabTransform
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import torch
from sklearn.preprocessing import LabelEncoder
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import Adam
import numpy as np
from utils import metrics, cost, safeCreateDir
from vision import plot_acc,plot_loss
import time

# 数据处理
def dataParse(text, stop_words):
    label,content,= text.split(' ')
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)

def getStopWords():
    file = open('./dataset/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

def getFormatData():
    file = open('./dataset/data/train1.txt', 'r',encoding='gbk')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in tqdm(texts, ncols=90):
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)

    ws = sum(all_words, [])
    set_ws = Counter(ws)
    keys = sorted(set_ws, key=lambda x: set_ws[x], reverse=True)
    dict_words = dict(zip(keys, list(range(1, len(set_ws) + 1))))
    ordered_dict = OrderedDict(dict_words)
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'])
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(all_words)
    vector = [torch.tensor(i) for i in vector]
    lengths = [len(i) for i in vector]
    pad_seq = pad_sequence(vector, batch_first=True)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(all_labels)
    data = pad_seq.numpy()
    num_classses = max(labels) + 1
    data = {'X': data,
            'label': labels,
            'num_classes': num_classses,
            'lengths': lengths,
            'num_words': len(my_vocab)}
    # print(data)
    io.savemat('./dataset/data/data.mat', data)

# 数据集加载
class Data(Dataset):
    def __init__(self, mode='train'):
        data = io.loadmat('./dataset/data/data.mat')
        self.X = data['X']
        self.y = data['label']
        self.lengths = data['lengths']
        self.num_words = data['num_words'].item()
        train_X, val_X, train_y, val_y, train_length, val_length = train_test_split(self.X, self.y.squeeze(), self.lengths.squeeze(),
                                                                                    test_size=0.3, random_state=1)
        val_X, test_X, val_y, test_y, val_length, test_length = train_test_split(val_X, val_y, val_length, test_size=0.5, random_state=2)
        if mode == 'train':
            self.X = train_X
            self.y = train_y
            self.lengths = train_length
        elif mode == 'val':
            self.X = val_X
            self.y = val_y
            self.lengths = val_length
        elif mode == 'test':
            # print('test_X',test_X)
            self.X = test_X
            self.y = test_y
            self.lengths = test_length
    def __getitem__(self, item):
        return self.X[item], self.y[item], self.lengths[item]
    def __len__(self):
        return self.X.shape[0]
class getDataLoader():
    def __init__(self,batch_size):

        train_data = Data('train')
        val_data = Data('val')
        test_data = Data('test')
        # print('test_data',test_data)
        self.traindl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valdl = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        self.testdl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        self.num_words = train_data.num_words



# 定义网络结构

class Model(nn.Module):
    def __init__(self, num_words, num_classes, input_size, hidden_dim):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(num_words, input_size)
        self.net = None
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

class GRU(Model):
    def __init__(self, num_words, num_classes, input_size=64, hidden_dim=32, num_layer=1):
        super(GRU, self).__init__(num_words, num_classes, input_size, hidden_dim)
        self.net = nn.GRU(input_size, hidden_dim, num_layer, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embeding(x)
        pd = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        output, hn = self.net(pd)
        pred = self.classification(hn[-1])
        return pred

# 定义训练过程



class Trainer():
    def __init__(self):
        safeCreateDir('results/')
        # self.args = args
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda')
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = getDataLoader(batch_size=64)
        self.traindl = data.traindl
        self.valdl = data.valdl
        self.testdl = data.testdl
        self.num_words = data.num_words

    def _init_model(self):
        self.net = None
        self.net = GRU(self.num_words, 15).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=5e-4)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        torch.save(self.net.state_dict(), 'results/gru.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load('results/gru.pt'))

    @torch.no_grad()
    def val(self):
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets, lengths) in enumerate(self.valdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to('cpu')
            pred = self.net(inputs, lengths)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        return acc, precision, f1, recall

    def train(self,epochs):
        patten = 'Iter: %d/%d   [===========]  cost: %.2fs  loss: %.4f  acc: %.4f/%.4f'
        train_accs = []
        val_accs = []
        c_loss = []
        for epoch in range(epochs):
            cur_preds = np.empty(0)
            cur_labels = np.empty(0)
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets, lengths) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                lengths1 = lengths.to('cpu')
                pred = self.net(inputs, lengths1)
                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
                cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
                cur_loss += loss.item()
            acc, precision, f1, recall = metrics(cur_preds, cur_labels)

            val_acc, val_precision, val_f1, val_recall = self.val()
            train_accs.append(acc)
            val_accs.append(val_acc)
            c_loss.append(cur_loss)
            end = time.time()
            print(patten % (
                epoch,
                epochs,
                end - start,
                cur_loss,
                val_acc,
                acc,
                # val_precision,
                # val_f1,
                # val_recall
            ))

        self.save_model()
        plot_acc(train_accs,val_accs)
        plot_loss(train_accs)

    @torch.no_grad()
    def test(self):
        print("test ...")

        self.load_model()
        patten = 'test score:  acc: %.4f   precision: %.4f   f1: %.4f   recall: %.4f'
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        print('self.testdl',self.testdl)
        for batch, (inputs, targets, lengths) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to('cpu')
            # print('inputs',inputs)
            pred = self.net(inputs, lengths)
            # print(pred)

            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        print(patten % (
            acc,
            precision,
            f1,
            recall
        ))





if __name__ == "__main__":
    # getFormatData()
    trainer=Trainer()
    trainer.train(epochs=50)
    trainer.test()