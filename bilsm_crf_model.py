from keras.layers import *
from keras_contrib.layers import CRF
import process_data
import pickle
from keras.models import *

EMBED_DIM = 200
BiRNN_UNITS = 200


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    #输入ID序列
    x_in = Input(shape=(None,))
    #构建词向量
    emb = Embedding(len(vocab), EMBED_DIM,mask_zero=True)(x_in)
    emb1 = Embedding(len(vocab), EMBED_DIM)(x_in)
    #使用CNN提取特征
    con1 = Conv1D(filters = BiRNN_UNITS, kernel_size = 1, activation='relu',padding = 'same')(emb1)
    con2 = Conv1D(filters = BiRNN_UNITS, kernel_size = 2, activation='relu',padding = 'same')(emb1)
    con3 = Conv1D(filters = BiRNN_UNITS, kernel_size = 3, activation='relu',padding = 'same')(emb1)
    con4 = Conv1D(filters = BiRNN_UNITS, kernel_size = 4, activation='relu',padding = 'same')(emb1)
    fc = concatenate([con1,con2,con3,con4])
    fc = Dense(BiRNN_UNITS, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    
    
    #使用BILSTM提取特征
    lstm = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(emb)
    
    #特征融合
    fc = concatenate([fc,lstm])
    #全连接
    fc = Dense(BiRNN_UNITS, activation='relu')(fc)
    #归一化
    fc = BatchNormalization()(fc)
    #CRF层预测
    crf = CRF(len(chunk_tags), sparse_target=True)
    outputs = crf(fc)
    model = Model([x_in],outputs)
    #打印模型图架构
    model.summary()
    #编译指定优化方法、损失函数以及监控的数据
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
