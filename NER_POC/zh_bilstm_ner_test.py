char_vocab_path = "./data/char_vocabs_zh.txt" # 字符字典文件
train_data_path = "./data/train_data600_final.txt" # 訓練數據(同列不可有空格，換下列需有空格)
test_data_path = "./data/test_data50_final.txt" # 測試數據

special_words = ['<PAD>', '<UNK>'] # 特殊詞表示

# "BIO"標記的標籤
label2idx = {'O':0,
                 'ADMISSIONDATE-B': 1,     #住院日期-開頭
                 'ADMISSIONDATE-I': 2,     #住院日期-以後
                 'DISCHARGEDATE-B': 3,     #出院日期-開頭
                 'DISCHARGEDATE-I': 4,     #出院日期-以後
                 'SURGERYDATE-B': 5,       #手術日期-開頭
                 'SURGERYDATE-I': 6,       #手術日期-以後
                 'OUTPATIENTDATE-B': 7,    #門診日期-開頭
                 'OUTPATIENTDATE-I': 8,    #門診日期-以後
                 'CHEMOTHERAPYDATE-B': 9,  #化療日期-開頭
                 'CHEMOTHERAPYDATE-I': 10, #化療日期-以後
                 'RADIOTHERAPYDATE-B': 11, #放療日期-開頭
                 'RADIOTHERAPYDATE-I': 12, #放療日期-以後
                 'DISEASE-B': 13,      #疾病症狀-開頭
                 'DISEASE-I': 14,      #疾病症狀-以後
                 'TREATMENT-B': 15,    #處置方式-開頭
                 'TREATMENT-I': 16,    #處置方式-以後
                 'BODY-B': 17,         #器官部位-開頭
                 'BODY-I': 18          #器官部位-以後
                 }

# 索引和BIO標籤對應
idx2label = {idx: label for label, idx in label2idx.items()}  #{ 0: 'O', 1: 'ADMISSION-B' }

# 讀取字符字典文件
with open(char_vocab_path, "r", encoding="utf-8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs

# 字符和索引編號對應
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)} #{0: '<PAD>', 1: '<UNK>', 2: '!', 3: '#', 4: '%', 5: '&'}
# print(len(idx2vocab))
# print(idx2vocab)
vocab2idx = {char: idx for idx, char in idx2vocab.items()} #{'<PAD>': 0, '<UNK>': 1, '!': 2, '#': 3, '%': 4, '&': 5}
# print(len(vocab2idx))

#-----------------------------------------------------------
# BUG 是簡繁轉換的問題, 重新確認數量
from collections import Counter

c = Counter(char_vocabs)
# print("char_vocabs中的數量：", c)

c_2 = Counter(c.values())
# print("重複的數量：", c_2)

# char_vocabs = list(set(char_vocabs))  # 去重複
char_vocabs = sorted(set(char_vocabs), key = char_vocabs.index)  # 去重複2
print("去重複後的char_vocabs個數：", len(char_vocabs))

# 重新生成 idx2vocab, vocab2idx
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}#{0: '<PAD>', 1: '<UNK>', 2: '!', 3: '#', 4: '%', 5: '&'}
# print(idx2vocab)
vocab2idx = {char: idx for idx, char in idx2vocab.items()} #{'<PAD>': 0, '<UNK>': 1, '!': 2, '#': 3, '%': 4, '&': 5}
# print(vocab2idx)

#========================================================

# 讀取訓練語料
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
#         print(lines[0:10])
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()  #strip( ):去除首尾空格 ; split():用空格分割
#             print(char)
#             print(label)
            sent_.append(char)
            tag_.append(label)
        else:
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
#             print(sent_ids)
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_ids)
            labels.append(tag_ids)
            sent_, tag_ = [], []
#             print(sent_)
    return datas, labels

# 加載訓練集
train_datas, train_labels = read_corpus(train_data_path, vocab2idx, label2idx)
# 加載測試集
test_datas, test_labels = read_corpus(test_data_path, vocab2idx, label2idx)

#---------------------------------------------------------------

import numpy as np
import keras

#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model
#from keras.utils.vis_utils import plot
from keras.utils import plot_model

from keras.models import Sequential
from keras.models import Model
from keras.layers import Masking, Embedding, Bidirectional, LSTM, Dense, Input, TimeDistributed, Activation, Dropout
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
K.clear_session()


#************
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

#*************

EPOCHS = 10
BATCH_SIZE = 64   # --> 128 or 64 or 32 --
EMBED_DIM = 300
HIDDEN_SIZE = 128
MAX_LEN = 200
VOCAB_SIZE = len(idx2vocab)
#VOCAB_SIZE = 6874
#VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)
print(VOCAB_SIZE, CLASS_NUMS)


print('padding sequences')
train_datas = sequence.pad_sequences(train_datas, maxlen=MAX_LEN)
train_labels = sequence.pad_sequences(train_labels, maxlen=MAX_LEN)
test_datas = sequence.pad_sequences(test_datas, maxlen=MAX_LEN)
test_labels = sequence.pad_sequences(test_labels, maxlen=MAX_LEN)
print('x_train shape:', train_datas.shape)
print('x_test shape:', test_datas.shape)

train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)
test_labels = keras.utils.to_categorical(test_labels, CLASS_NUMS)
print('trainlabels shape:', train_labels.shape)
print('testlabels shape:', test_labels.shape)

## BiLSTM+CRF 模型建構

inputs = Input(shape=(MAX_LEN,), dtype='int32')
x = Masking(mask_value=0)(inputs)
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(x)

x = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(x)
x = Dropout(rate=0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(rate=0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(rate=0.2)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(rate=0.2)(x)

x = Dense(64, activation = "tanh")(x)

x = TimeDistributed(Dense(CLASS_NUMS))(x)

outputs = CRF(CLASS_NUMS)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


#********加入history********
#創建一個實例history
history = LossHistory()
#********加入history********


model.compile(loss=crf_loss, optimizer='Adam', metrics=[crf_accuracy])
model.fit(train_datas, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=[history])

score = model.evaluate(test_datas, test_labels, batch_size=BATCH_SIZE)
print(model.metrics_names)
print(score)

# import pydotplus
# import keras.utils
# keras.utils.vis_utils.pydot = pydotplus
# keras.utils.plot_model(model, to_file='model_plot_crf_v600_final_ed.png', show_shapes=True, show_layer_names=True)  # keras model Image

# save model
# model.save("./model/ch_ner_model_crf_v600_final_ed.h5")