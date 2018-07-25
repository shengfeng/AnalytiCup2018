# coding: utf-8

# In[1]:


import sys
import collections
import mxnet as mx
from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.contrib import text
import os
import random
import zipfile
from sklearn.model_selection import train_test_split
import spacy
import time
from time import strftime

# In[2]:


ROOT_PATH = "data/"

# In[3]:


es_e_l = []
es_s_l = []
es_e_r = []
es_s_r = []
es_labels = []
## english-spanish text
with open(os.path.join(ROOT_PATH, "cikm_english_train_20180516.txt"), 'r', encoding='utf-8') as esf:
    for line in esf:
        segs = line.strip().replace('?', '').split('\t')
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
with open(os.path.join(ROOT_PATH, "cikm_spanish_train_20180516.txt"), 'r', encoding='utf-8') as ssf:
    for line in ssf:
        segs = line.strip().replace('?', '').split('\t')
        se_e_l.append(segs[0].lower())
        se_e_r.append(segs[2].lower())
        se_s_l.append(segs[1].lower())
        se_s_r.append(segs[3].lower())
        se_labels.append(int(segs[4]))

test_s_1 = []
test_s_2 = []
test_label = []
## spanish test file
with open(os.path.join(ROOT_PATH, "test_300_label.txt"), 'r', encoding='utf-8') as tef:
    for line in tef:
        segs = line.strip().replace('?', '').replace('¿', '').split('\t')
        test_s_1.append(segs[0].lower())
        test_s_2.append(segs[1].lower())
        test_label.append(int(segs[2]))

print("es data size:", len(es_s_l))
print("se data size:", len(se_e_l))
print("test data size:", len(test_s_1))

# In[4]:


##add data sets (s0, s1, y) + (s1, s0, y)
left_texts = es_s_l + se_s_l
right_texts = es_s_r + se_s_r
y = es_labels + se_labels
print("left data size:", len(left_texts))
print("right data size:", len(right_texts))
print("label size:", len(y))

# In[5]:


spacy_en = spacy.load('en')
spacy_es = spacy.load('es')


def en_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def es_tokenizer(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]


# In[6]:


left_tokenized = []
right_tokenized = []

for token in left_texts:
    left_tokenized.append(es_tokenizer(token))

for token in right_texts:
    right_tokenized.append(es_tokenizer(token))

token_counter = collections.Counter()
for sample in left_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = 1
        else:
            token_counter[token] += 1

for sample in right_tokenized:
    for token in sample:
        if token not in token_counter:
            token_counter[token] = 1
        else:
            token_counter[token] += 1


# In[7]:


# 根据词典，将数据转换成特征向量。
def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=500, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）。
            while len(padded_feature) < maxlen:
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features


# In[8]:


vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>', reserved_tokens=None)

left_texts_features = encode_samples(left_tokenized, vocab)
right_texts_featrues = encode_samples(right_tokenized, vocab)

left_padded_features = pad_samples(left_texts_features, 40, 0)
right_texts_featrues = pad_samples(right_texts_featrues, 40, 0)

# In[9]:


num_validation_samples = int(0.2 * len(left_padded_features))

left_x_train = left_padded_features[:-num_validation_samples]
left_x_val = left_padded_features[-num_validation_samples:]

right_x_train = right_texts_featrues[:-num_validation_samples]
right_x_val = right_texts_featrues[-num_validation_samples:]

y_train = y[:-num_validation_samples]
y_val = y[-num_validation_samples:]

# In[10]:


inputs = []
for i in range(len(left_x_train)):
    inputs.append([left_x_train[i], right_x_train[i]])

val_inputs = []
for i in range(len(left_x_val)):
    val_inputs.append([left_x_val[i], right_x_val[i]])

# In[11]:


test_left_tokenized = []
test_right_tokenized = []

for token in test_s_1:
    test_left_tokenized.append(es_tokenizer(token))

for token in test_s_2:
    test_right_tokenized.append(es_tokenizer(token))

test_left_features = encode_samples(test_left_tokenized, vocab)
test_right_featrues = encode_samples(test_right_tokenized, vocab)

test_left_padded_features = pad_samples(test_left_features, 40, 0)
test_right_texts_featrues = pad_samples(test_right_featrues, 40, 0)

test_inputs = []
for i in range(len(test_left_padded_features)):
    test_inputs.append([test_left_padded_features[i], test_right_texts_featrues[i]])

# In[12]:


ctx = mx.gpu()

train_features = nd.array(inputs, ctx=ctx)
train_label = nd.array(y_train, ctx)
val_features = nd.array(val_inputs, ctx=ctx)
val_label = nd.array(y_val, ctx)
test_featrues = nd.array(test_inputs, ctx=ctx)
test_label = nd.array(test_label, ctx=ctx)
# In[13]:


glove_embedding = text.embedding.CustomEmbedding(pretrained_file_path='data/wiki.es.vec', vocabulary=vocab)


# In[14]:


# 相似度计算
def exponent_neg_manhattan_distance(left, right):
    return nd.exp(-nd.sum(nd.abs(left - right), axis=1, keepdims=True))


# new metric
def logloss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# In[15]:


class SentimentNet(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(len(vocab), embed_size)

            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.dense0 = nn.Dense(256, activation='relu', flatten=False)
            self.norm2 = nn.BatchNorm()
            self.dense1 = nn.Dense(128, activation='relu', flatten=False)
            self.norm3 = nn.BatchNorm()
            self.dropout = nn.Dropout(0.5)
            self.decoder = nn.Dense(num_outputs, flatten=False, activation='sigmoid')

    def forward(self, inputs):
        inputs = nd.transpose(inputs, axes=(1, 2, 0))

        bembeddings = self.embedding(inputs[0])
        bstates = self.encoder(bembeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        bencoding = nd.concat(bstates[0], bstates[-1])
        bencoding = self.dense0(bencoding)

        eembeddings = self.embedding(inputs[1])
        estates = self.encoder(eembeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        eencoding = nd.concat(estates[0], estates[-1])
        eencoding = self.dense0(eencoding)

        y = exponent_neg_manhattan_distance(bencoding, eencoding)
        #         print('the shape of y: ', y.shape)
        outputs = self.norm2(y)
        outputs = self.dense1(outputs)
        outputs = self.norm3(outputs)
        outputs = self.dropout(outputs)
        outputs = self.decoder(outputs)

        return outputs


# In[21]:


num_outputs = 1
lr = 0.01
num_epochs = 20
batch_size = 200
embed_size = 300
num_hiddens = 100
num_layers = 3
bidirectional = True

# In[22]:


net = SentimentNet(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)
# 设置 embedding 层的 weight 为预训练的词向量。
net.embedding.weight.set_data(glove_embedding.idx_to_vec.as_in_context(ctx))
# 训练中不更新词向量（net.embedding中的模型参数）。
net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)


# In[37]:


def eval_model(features, labels):
    l_sum = 0
    for i in range(features.shape[0] // batch_size):
        X = features[i * batch_size: (i + 1) * batch_size].as_in_context(ctx)
        y = labels[i * batch_size:(i + 1) * batch_size].as_in_context(ctx)
        output = net(X)
        l = loss(output, y)
        l_sum += l.sum().asscalar()
    return l_sum / features.shape[0]


# In[23]:


model_file = "Model_" + strftime("%Y-%m-%d %H-%M", time.localtime()) + ".mdl"

for epoch in range(1, num_epochs + 1):
    previous_train_loss = 1.0
    for i in range(train_features.shape[0] // batch_size):
        X = train_features[i * batch_size: (i + 1) * batch_size].as_in_context(ctx)
        y = train_label[i * batch_size: (i + 1) * batch_size].as_in_context(ctx)
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    train_loss = eval_model(train_features, train_label)
    val_loss = eval_model(val_features, val_label)
    test_loss = eval_model(test_featrues, test_label)
    if train_loss < previous_train_loss:
        net.save_params(model_file)
    print('epoch %d, train loss %.6f, val loss %.6f, test_loss %.6f' % (epoch, train_loss, val_loss, test_loss))


# In[39]:


def predict(features):
    output = []
    for i in range(features.shape[0] // batch_size):
        X = features[i * batch_size: (i + 1) * batch_size].as_in_context(ctx)
        y = net(X).asnumpy().flatten()
        for i in y:
            output.append(i)

    return output


# In[40]:

test_online_1 = []
test_online_2 = []
## spanish test file
with open(os.path.join(ROOT_PATH, "cikm_test_a_20180516.txt"), 'r', encoding='utf-8') as tef:
    for line in tef:
        segs = line.strip().replace('?', '').replace('¿', '').split('\t')
        test_online_1.append(segs[0].lower())
        test_online_2.append(segs[1].lower())

test_online_left_tokenized = []
test_online_right_tokenized = []

for token in test_online_1:
    test_online_left_tokenized.append(es_tokenizer(token))

for token in test_online_2:
    test_online_right_tokenized.append(es_tokenizer(token))

test_online_left_features = encode_samples(test_online_left_tokenized, vocab)
test_online_right_featrues = encode_samples(test_online_right_tokenized, vocab)

test_online_left_padded_features = pad_samples(test_online_left_features, 40, 0)
test_online_right_texts_featrues = pad_samples(test_online_right_featrues, 40, 0)

test_online_inputs = []
for i in range(len(test_online_left_padded_features)):
    test_online_inputs.append([test_online_left_padded_features[i], test_online_right_texts_featrues[i]])

test_online_featrues = nd.array(test_online_inputs, ctx=ctx)

output = predict(test_online_featrues)

# In[28]:


output

# In[43]:

result_file_name = strftime("%Y-%m-%d %H-%M", time.localtime()) + "result.txt"

with open(result_file_name, 'w') as f:
    for line in output:
        f.write(str(round(line, 6)) + '\n')

