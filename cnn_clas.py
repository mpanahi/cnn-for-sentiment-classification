from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Input,Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras import backend as K
from sklearn.feature_extraction.text import *
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D

max_features = 20000
# max_features = 2000000000000000000000000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
maxlen = 21683
batch_size = 32
# from theano import function

print("Loading data...")
corpus = []
test = []
validation = []
train_lables = []
test_lables = []
validation_lables = []
c = 0
f1 = open('negative_cmts_final', encoding='utf8')
for file in iter(f1):
    corpus.append(file)

    train_lables.append(-1)
    c = c + 1
f1.close()

f2 = open('positive_cmts_final', encoding='utf8')
for file in iter(f2):
    corpus.append(file)

    train_lables.append(1)
    c = c + 1
f2.close()


test_data=[]
g = open('nehative',encoding='utf8')
i=1
test_lables=[]
for file in iter(g):
    test_data.append(file)
    test_lables.append(-1)
g.close()
g = open('positive',encoding='utf8')
for file in iter(g):
    test_data.append(file)
    test_lables.append(1)
g.close()
vectorizer = CountVectorizer()
train_vectors = []
print(train_lables)
sh = []
nb_epochs = 10
batch_size = 64
train_vectors = vectorizer.fit_transform(corpus)
test_vectors = vectorizer.transform(test_data)
print("features")
print(vectorizer.get_feature_names()[2])
z=test_vectors
sh = []
# test_vectors = vectorizer.transform(test)
# validation_vectors = vectorizer.transform(valid
# print(train_vectors.shape[0])
import numpy as np

x_train = train_vectors
# x_test = test_vectors
# x_validation = validation_vectors
# print(x_train[0])
print(x_train.shape)

X_train = x_train.todense()
z = z.todense()
#X_test=z.todense()
print("12345")
print(X_train.shape)
# a=[1,2,3]
# print(a.shape)
# X_validation=x_validation.todense()
ls1=[]
ls2=[]
for i in range(0,2):
    ls2.append(0)
for t in train_lables:
    if t==1:
        ls2 = []
        for i in range(0, 2):
            ls2.append(0)
        for i in range(0,2):
            if i==0:
                ls2[i]=1
            else:
                ls2[i] = 0
        ls1.append(ls2)
    if t==-1:
        ls2 = []
        for i in range(0, 2):
            ls2.append(0)
        for i in range(0,2):
            if i==1:
                ls2[i]=1
            else:
                ls2[i] = 0
        ls1.append(ls2)


y_train = ls1
embedding_layer = Embedding(29285 + 1,
                            100,

                            input_length=29285,
                            trainable=True)
#model.add(Embedding(38946, 100, input_length = 258))
#print(model.layers[0].get_weights()[0].shape)
convs = []
filter_sizes = [1,2,3]
sequence_input = Input(shape=(29285,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
print("this this this")
print(embedded_sequences)
for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
l_merge = Merge(mode='concat', concat_axis=1)(convs)
print(l_merge.shape)
#l_cov1= Conv1D(128, 1, activation='relu')(l_merge)
#l_pool1 = MaxPooling1D(1)(l_cov1)
#l_cov2 = Conv1D(128, 1, activation='relu')(l_pool1)
#l_pool2 = MaxPooling1D(1)(l_cov2)
l_flat = Flatten()(l_merge)
print(l_flat.shape)
l_dense = Dense(128, activation='relu')(l_flat)
d=Dropout(0.5)(l_dense)
preds = Dense(2, activation='softmax')(d)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
#model.summary()
#model.fit(train_data, train_lables,
         #nb_epoch=2, batch_size=50)
#model.save('cnn_with_embedding.h5')
from keras.models import load_model
#model1=load_model('cnn_with_embedding.h5')
#model1=load_model('cnn_with_embedding.h5')
#model1.compile(loss='binary_crossentropy',
              #optimizer = 'adam',
             # metrics=["accuracy"])
model.fit(X_train, y_train,
          nb_epoch=5, batch_size=50)
a=model.predict(z,batch_size=50)
from sklearn.externals import joblib
joblib.dump(a, 'results_cnn_now.joblib')
#print(a)
#for b in a:
    #print(b)
#print(a.shape)
#model1=model.save('cnn_without_pretraining_sentiment_insta.h5')
#model1=load_model('cnn_with_embedding.h5')
