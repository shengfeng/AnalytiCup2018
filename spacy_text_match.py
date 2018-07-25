
# coding: utf-8

# ### Read sentences

# In[1]:


import os
import random
import time
from time import strftime

from keras import backend as K

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, LSTM
from keras.layers import Embedding, Dropout, Bidirectional
from keras.models import Model, Sequential
from keras.layers import Activation, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
import keras

import spacy
import gc


# ### Read text data

# In[2]:


ROOT_PATH  = ""


# In[3]:


es_e_l = []
es_s_l = []
es_e_r = []
es_s_r = []
es_labels = []
## english-spanish text 
with open(os.path.join(ROOT_PATH, "data/cikm_english_train_20180516.txt"), 'r', encoding='utf-8') as esf:
    for line in esf:
        segs = line.strip().replace('?','').replace('¿', '').replace(',', '').replace('.', '').split('\t')
        es_e_l.append(segs[0].lower())
        es_e_r.append(segs[2].lower())
        es_s_l.append(segs[1].lower())
        es_s_r.append(segs[3].lower())
        es_labels.append(int(segs[4]))
        
se_e_l = []
se_s_l = []
se_e_r = []
se_s_r = []
se_labels = []
## spanish-english text
with open(os.path.join(ROOT_PATH, "data/cikm_spanish_train_20180516.txt"), 'r', encoding='utf-8') as ssf:
    for line in ssf:
        segs = line.strip().replace('?','').replace('¿', '').replace(',', '').replace('.', '').split('\t')
        se_s_l.append(segs[0].lower())
        se_s_r.append(segs[2].lower())
        se_e_l.append(segs[1].lower())
        se_e_r.append(segs[3].lower())
        se_labels.append(int(segs[4]))

test_s_1 = []
test_s_2 = []
## spanish test file
with open(os.path.join(ROOT_PATH, "data/cikm_test_a_20180516.txt"), 'r', encoding='utf-8') as tef:
    for line in tef:
        segs = line.strip().replace('?','').replace('¿', '').replace(',', '').replace('.', '').split('\t')
        test_s_1.append(segs[0].lower())
        test_s_2.append(segs[1].lower())

print("es data size:", len(es_s_l))
print("se data size:", len(se_e_l))
print("test data size:", len(test_s_1))


# In[4]:


##add data sets (s0, s1, y) + (s1, s0, y)
left_texts = es_s_l + se_s_l 
right_texts = es_s_r + se_s_r
y = es_labels + se_labels
#left_texts += se_s_r + es_s_r
#right_texts += es_s_l + se_s_l 
#y += se_labels + es_labels
print("left data size:", len(left_texts))
print("right data size:", len(right_texts))
print("label size:", len(y))


# In[5]:


span_sw = []
with open(os.path.join(ROOT_PATH, "data/spanish.txt"), 'r', encoding='utf-8') as swf:
    for line in swf:
        word = line.strip()
        span_sw.append(word)


# ### Spacy tokenizer 

# In[6]:


spacy_es = spacy.load('es')


# In[7]:


def es_tokenizer(text):
    return [tok.text for tok in spacy_es.tokenizer(text) if tok.text not in span_sw ]

def texts_to_sequences(tokenized_texts, token_counter):
    sequences = []
    for texts in tokenized_texts:
        sub_seq = []
        for token in texts:
            sub_seq.append(token_counter[token])
        sequences.append(sub_seq)
    return sequences


# In[8]:


## train text data
left_tokenized = []
right_tokenized = []
## test text data
test_left_tokenized = []
test_right_tokenized = []

for token in left_texts:
    left_tokenized.append(es_tokenizer(token))

for token in right_texts:
    right_tokenized.append(es_tokenizer(token))

for token in test_s_1:
    test_left_tokenized.append(es_tokenizer(token))

for token in test_s_2:
    test_right_tokenized.append(es_tokenizer(token))

## form wrods dictionary {word: index} 
token_counter = {}
index = 0
for sample in left_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = index
            index += 1

for sample in right_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = index
            index += 1

for sample in test_left_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = index
            index += 1
            
for sample in test_right_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = index
            index += 1
print("unique token count: ", len(token_counter))


# ### Load word_vec

# In[9]:


## es.vec
es_vec = {}
with open(os.path.join(ROOT_PATH, "data/wiki.es.vec"), 'r', encoding='utf-8') as vecf:
    i = 0
    for line in vecf:
        if i == 0:
            i = 1
            continue
        segs = line.strip().split(' ')
        es_vec[segs[0]] = ' '.join(segs[1:])


# In[10]:


L_MAX_SEQUENCE_LENGTH = 40  #左边最大句子长度
R_MAX_SEQUENCE_LENGTH = 40  #右边最大句子长度
MAX_SEQUENCE_LENGTH = 40    #平均最大句子长度
MAX_NB_WORDS = 20000      #词典大小，词的个数
EMBEDDING_DIM = 300       #词向量维度
VALIDATION_SPLIT = 0.2    # 测试集比例
BATCH_SIZE = 100      ##batch大小
EPOCH = 60  ## 迭代次数


# In[11]:


# ## Fit tokenizer
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(left_texts + right_texts + test_s_1 + test_s_2)


# In[12]:


# prepare embedding matrix
#word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(token_counter))
num_words = min(MAX_NB_WORDS, len(token_counter))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
print('Preparing embedding matrix. :', embedding_matrix.shape)
for word, i in token_counter.items():
    embedding_vector = es_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = list(map(eval, embedding_vector.split(' ')))
del es_vec
gc.collect()


# In[13]:


# prepare left data
# sequences_l = tokenizer.texts_to_sequences(left_texts)
sequences_l = texts_to_sequences(left_tokenized, token_counter)
MAX_LENGTH = 0
for s in sequences_l:
    if len(s) > MAX_LENGTH:
        MAX_LENGTH = len(s)
print(MAX_LENGTH)
left_data = pad_sequences(sequences_l, maxlen=L_MAX_SEQUENCE_LENGTH)
print('Shape of left_data tensor:', left_data.shape)


# In[14]:


# prepare right data
# sequences_r = tokenizer.texts_to_sequences(right_texts)
sequences_r = texts_to_sequences(right_tokenized, token_counter)
MAX_LENGTH = 0
for s in sequences_r:
    if len(s) > MAX_LENGTH:
        MAX_LENGTH = len(s)
print(MAX_LENGTH)
right_data = pad_sequences(sequences_r, maxlen=R_MAX_SEQUENCE_LENGTH)
print('Shape of right_data tensor:', right_data.shape)


# In[15]:


## split train and val sets
random.seed(7)
indices = np.arange(left_data.shape[0])
np.random.shuffle(indices)
data_1 = left_data[indices]
y = np.array(y)
labels = y[indices]
num_validation_samples = int(VALIDATION_SPLIT * data_1.shape[0])
print(num_validation_samples)

left_x_train = data_1[:-num_validation_samples]
left_x_val = data_1[-num_validation_samples:]

data_2 = right_data[indices]
right_x_train = data_2[:-num_validation_samples]
right_x_val = data_2[-num_validation_samples:]

y_train = labels[:-num_validation_samples]
y_val = labels[-num_validation_samples:]


# In[16]:


## prepare test data
sequences_test_1 = texts_to_sequences(test_left_tokenized, token_counter)
test_1 = pad_sequences(sequences_test_1, maxlen=L_MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_1.shape)

sequences_test_2 = texts_to_sequences(test_right_tokenized, token_counter)
test_2 = pad_sequences(sequences_test_2, maxlen=R_MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_2.shape)


# ### import custom test data

# In[17]:


test_new_1 = []
test_new_2 = []
label_300 = []
## spanish test file
with open(os.path.join(ROOT_PATH, "data/test_300_label.txt"), 'r', encoding='utf-8') as newf:
    for line in newf:
        segs = line.strip().replace('?','').replace('¿', '').replace(',', '').replace('.', '').split('\t')
        test_new_1.append(segs[0].lower())
        test_new_2.append(segs[1].lower())
        label_300.append(int(segs[2]))
label_300 = np.array(label_300)


# In[18]:


test_300_left_tokenized = []
test_300_right_tokenized = []
for token in test_new_1:
    test_300_left_tokenized.append(es_tokenizer(token))

for token in test_new_2:
    test_300_right_tokenized.append(es_tokenizer(token))


# In[19]:


print(test_300_left_tokenized[0])
print(test_300_right_tokenized[0])


# In[20]:


## prepare test 300 data
sequences_test_300_1 = texts_to_sequences(test_300_left_tokenized, token_counter)
test_300_1 = pad_sequences(sequences_test_300_1, maxlen=L_MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_300_1.shape)

sequences_test_300_2 = texts_to_sequences(test_300_right_tokenized, token_counter)
test_300_2 = pad_sequences(sequences_test_300_2, maxlen=R_MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_300_2.shape)


# ### Build model

# In[30]:


embedding_layer = Embedding(num_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)


# In[31]:


lstm_layer = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2), merge_mode='sum')


# In[32]:


input1 = keras.layers.Input(shape=(L_MAX_SEQUENCE_LENGTH,), dtype='float32')
left_embedded_sequences = embedding_layer(input1)
x1 = lstm_layer(left_embedded_sequences)
#x1 = Dropout(0.5)(x1)
#x1 = BatchNormalization()(x1)
#x1 = Dense(64, activation='relu')(x1) ## acitivation = tanh, relu, sigmoid


# In[33]:


input2 = keras.layers.Input(shape=(R_MAX_SEQUENCE_LENGTH,), dtype='float32')
right_embedded_sequences = embedding_layer(input2)
x2 = lstm_layer(right_embedded_sequences)
#x2 = Dropout(0.5)(x2)
#x1 = BatchNormalization()(x1)
#x2 = Dense(64, activation='relu')(x2) ## acitivation = tanh, relu, sigmoid


# In[34]:


# new metric
def logloss(y_true, y_pred):
    return -K.mean(y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred))


# In[ ]:


merged = keras.layers.subtract([x1, x2])  # add, concatenate, maximum
merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
merged = Dense(100, activation='relu')(merged)
merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
merged = Dense(50, activation='relu')(merged)
merged = BatchNormalization()(merged)

# model_file = "Model_" + strftime("%Y-%m-%d %H-%M", time.localtime()) + ".mdl"
# model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

output = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=[logloss])  ## optimizer= sgd, adam, rmsprop
# early_stopping = EarlyStopping(monitor='val_loss ', patience=10, mode='min')
model.fit([left_x_train, right_x_train], y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=([test_300_1, test_300_2], label_300), verbose=1)


# In[ ]:


score = model.evaluate([test_300_1, test_300_2], label_300, batch_size=BATCH_SIZE, verbose=1)
predicts = model.predict([test_1, test_2], batch_size=BATCH_SIZE, verbose=1)


# In[ ]:


test_predicts = model.predict([test_1, test_2], batch_size=BATCH_SIZE, verbose=1)
re_f = open('data/result_0.500.txt', 'w')
for i in range(len(test_1)):
    pre = predicts[i][0]
    re_f.write(str(round(pre, 6)) + '\n')
re_f.close()


# In[ ]:


import math
def cal_log_loss(y, p):
    result = 0
    for i in range(len(y)):
        result += y[i]*math.log(p[i]) + (1-y[i])*math.log(1-p[i])
    return -1 * result / len(y)


# In[ ]:


predicts_300 = model.predict([test_300_1, test_300_2], batch_size=BATCH_SIZE, verbose=1)
p = []
for i in range(len(predicts_300)):
    pre = predicts_300[i][0]
    p.append(round(pre, 6))


# In[ ]:


log_loss = cal_log_loss(label_300, p)
log_loss


# In[ ]:


from keras.models import load_model
model = load_model('Model_2018-07-17 02-15.mdl', {'logloss': logloss})

