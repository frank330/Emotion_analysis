# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import  Dense, Dropout, Input, Embedding,SimpleRNN,LSTM
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence
from jieba import lcut
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping
import PySimpleGUI as sg

# 数据处理
# """判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str
def getStopWords():
    file = open('./data/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words
def dataParse(text, stop_words):
    label, content, = text.split('	####	')
    # label = label_map[label]
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)

def getData(file='./data/data.txt',):
    file = open(file, 'r',encoding='gbk')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)
    return all_words,all_labels


## 读取测数据集
data,label = getData()

X_train, X_t, train_y, v_y = train_test_split(data,label,test_size=0.3, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t,v_y,test_size=0.5, random_state=42)
# print(X_train)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(np.array(train_y).reshape(-1,1)).toarray()
val_y = ohe.transform(np.array(val_y).reshape(-1,1)).toarray()
test_y = ohe.transform(np.array(test_y).reshape(-1,1)).toarray()

## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 100
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(data)
# texts_to_sequences 输出的是根据对应关系输出的向量序列，是不定长的，跟句子的长度有关系
train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)
## 将每个序列调整为相同的长度.长度为100
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
num_classes = 2
## 定义LSTM模型
inputs = Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(layer)
layer = Dense(128,activation="relu",name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #模型训练
model.fit(train_seq_mat,train_y,batch_size=128,epochs=10,
                          validation_data=(val_seq_mat,val_y),
                          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)] ## 当val-loss不再提升时停止训练
                         )
    # 保存模型
model.save('model/LSTM.h5')
del model
## 对验证集进行预测
    # 导入已经训练好的模型
model = load_model('model/LSTM.h5')

test_pre = model.predict(test_seq_mat)
pred = np.argmax(test_pre,axis=1)
real = np.argmax(test_y,axis=1)
cv_conf = confusion_matrix(real, pred)
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')
patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
print(patten % (acc,precision,recall,f1,))


def dataParse_(content, stop_words):
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words
def getData_one(text):
    stop_words = getStopWords()
    all_words = []
    content = dataParse_(text, stop_words)
    all_words.append(content)
    return all_words

def predict_(text_o):
    data_cut = getData_one(text_o)
    t_seq = tok.texts_to_sequences(data_cut)
    t_seq_mat = sequence.pad_sequences(t_seq, maxlen=max_len)
    model = load_model('model/LSTM.h5')
    t_pre = model.predict(t_seq_mat)
    pred = np.argmax(t_pre, axis=1)
    labels11 = ['negative', 'active']
    pred_lable = []
    for i in pred:
        pred_lable.append(labels11[i])
    return pred_lable[0]

def main_windows():

    # 菜单栏
    menu_def = [['Help', ['About...', ['你好']]], ]

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [sg.Menu(menu_def, tearoff=True)],
              [sg.Text('')],
              [sg.Multiline(s=(60, 20), key='_INPUT_news_', expand_x=True)],
              [sg.Text('')],
              [sg.Text('', s=(12)), sg.Text('分析结果：', font=("Helvetica", 15)),
               sg.Text('     ', key='_OUTPUT_news_', font=("Helvetica", 15))],
              [sg.Text('')],
              [sg.Text('', s=(12)), sg.Button('开始', font=("Helvetica", 15)), sg.Text('', s=(10)),
               sg.Button('清空', font=("Helvetica", 15)),
               sg.Text('', s=(4))],
              [sg.Text('')],
              [sg.Sizegrip()]
              ]

    window = sg.Window('情感分析系统', layout,
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window.set_min_size(window.size)



    while True:
        event, values = window.read(timeout=100)

        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break
        elif event == '开始':
            kk = predict_(values['_INPUT_news_'])
            window['_OUTPUT_news_'].update(kk)
        elif event == '清空':
            window['_OUTPUT_news_'].update(' ')
            window['_INPUT_news_'].update('')

    window.close()
    exit(0)


if __name__ == "__main__":



    main_windows()
