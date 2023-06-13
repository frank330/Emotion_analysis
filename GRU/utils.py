from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import time
import os


def metrics(pred, real):
    precision = precision_score(pred, real, average='micro')
    acc = accuracy_score(pred, real)
    f1 = f1_score(pred, real, average='micro')
    recall = recall_score(pred, real, average='micro')
    return acc, precision, f1, recall


def cost(fun):
    def use_time(*arg, **args):
        start = time.time()
        fun(*arg, **args)
        end = time.time()
        print('cost: %.2s' % (end-start))

    return use_time


def safeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)