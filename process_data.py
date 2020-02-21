import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import random

def load_data():

    train = _parse_data(open('medical_data/train_data', 'rb'))
    test = _parse_data(open('medical_data/test_data', 'rb'))
   
    #对词进行统计
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    #根据词构建词典，仅仅保留个数大于等于2的词
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    #构建实体标识列表
    chunk_tags = ['O', 'B-ILL', 'I-ILL', 'B-MED', 'I-MED', "B-PRE", "I-PRE",
                  "B-PUL","I-PUL","B-SYM","I-SYM","B-SYN","I-SYN","B-TIM","I-TIM"]
    #存储词典，实体标识列表
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)


    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)
'''

    输入：文件路径
    输出：三维列表，每个句子 句子每个词和其对应的实体标识

    [[[词,实体标识（B_ILL等）],[],[]...]]
'''

def _parse_data(fh):

    #windows和linux系统对回车换行使用了不同的符号
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    #读取文档所有数据
    string = fh.read().decode('utf-8')
    #返回的结果
    data = list()

    #首先对每个句子进行分割
    for sample in string.strip().split(split_text + split_text):
        tmp = list()
        flag = False
        #句子中的词进行分割
        for row in sample.split(split_text):
            word_tmp = row.strip().split()
            if(len(word_tmp)==2):
                #对存在实体的数据进行保留
                if("B" in word_tmp[1]):
                    flag = True
                tmp.append(word_tmp)
        if(flag):
            data.append(tmp)
    return data

'''
    输入：原始数据，词典，实体标识列表，最大长度，是否使用one-hot
    输出：根据词典将原始数据词转换成ID值，根据实体标识列表将原始数据标识转换成ID值
'''

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen) 
    return x, length
