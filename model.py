import pandas as pd
from scipy.sparse import data
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GRU, Lambda, Dense, Bidirectional
import tensorflow.keras.backend as K
import warnings
import os

warnings.filterwarnings('ignore')

def read_fasta_file(x, number_sample=9472):
    seq = [0] * (number_sample)
    id = []
    with open(x, encoding='utf8') as f:
        for line in f:
            if line.startswith('>'):
                tmp = ''
                id.append(line)
            else:
                tmp = tmp + line.replace('\n', '')
            seq[len(id) - 1] = tmp
    data = pd.DataFrame()
    data['id'] = id
    data['seq'] = seq
    return data

train = read_fasta_file('./训练集/astral_train.fa', 9472)
test = read_fasta_file('./测试集/astral_test.fa', 2371)


# 训练数据提取标签
train['label'] = train['id'].apply(lambda x: str(x).split(' ')[1])
train['id'] = train['id'].apply(lambda x: str(x).split(' ')[0].replace('>', ''))
train['label'] = train['label'].apply(lambda x: '.'.join(str(x).split('.')[:2]))
# 245 分类，多分类任务
test['id'] = test['id'].apply(lambda x: str(x).replace('>', '').strip())

# label转数值
train_label = list(train['label'].unique())
label_2_number = dict(zip(train_label, list(range(0, len(train_label)))))
number_2_number = dict(zip(list(range(0, len(train_label))), train_label))
train['label'] = train['label'].map(label_2_number)

# 字符串转列表
train['seq_list'] = train['seq'].apply(lambda x: list(x))
test['seq_list'] = test['seq'].apply(lambda x: list(x))

# 词向量参数
embed_size = 128
MAX_NB_WORDS = 128
MAX_SEQUENCE_LENGTH = 256
Class_Type = len(label_2_number)

X_train = train['seq_list']
X_test = test['seq_list']

# 创建词典，参数num_words 为词典的频次最大值
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
# 该语料中单词的总数 max_features
nb_words = len(word_index) + 1

# 数据切分
x = X_train
y = train['label'].values
trn_x, val_x, trn_y, val_y = train_test_split(x, y, random_state=1, test_size=0.1, stratify=y)
# label转换01编码
cat_trn_y = to_categorical(trn_y, Class_Type)
cat_val_y = to_categorical(val_y, Class_Type)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 模型构造
model = Sequential()
model.add(Embedding(nb_words, 128, input_length=MAX_SEQUENCE_LENGTH,mask_zero=False))
model.add(keras.layers.Conv1D(256, 37, padding='same', strides=1,activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=36,padding='same'))
model.add(keras.layers.Flatten())
model.add(Dense(Class_Type,activation='softmax'))
# 模型配置
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#早停函数
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0, mode='max', restore_best_weights=True)
#模型训练
model.fit(trn_x,cat_trn_y,validation_data=(val_x,cat_val_y
),batch_size=32,epochs=100,callbacks=[early_stop],shuffle=False)


# 结果导出
predictions = model.predict_classes(X_test)
result = pd.DataFrame()
result['sample_id'] = test['id'].copy()
result['category_id'] = predictions
result['category_id'] = result['category_id'].map(number_2_number)

result.to_csv('res5.csv',index=None)