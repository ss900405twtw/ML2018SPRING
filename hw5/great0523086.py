import numpy as np 
import pandas as pd 
import os
import csv 
import re
from keras.layers import *
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten,Dense, Dropout, Activation
import _pickle as pk
import keras
import sys




f = open(sys.argv[2],'r')  
result = list()  

for line in open(sys.argv[2]):  
    line = f.readline()  
    
    result.append(line) 

# print(train)
# print(len(result))
result2=np.array(result).reshape(len(result),1)
result2=[result2[i][0].split('\n')[0] for i in range(len(result2))]
#print(result2)
raw_x=result2
#print(raw_x[0].split('\n')[0])
#S=[raw_x[i].split()[0] for i in range(len(raw_x))]
#print(S)
result=[result[i].split('\n')[0] for i in range(len(raw_x))]
#print(result)
# xxx=
#print(raw_x[-1])
#xxx=np.array(result)
xxx=result2
# print(xxx)
#print(xxx)

S=[xxx[i].split() for i in range(len(raw_x))]
#print(S)
xx=xxx



# parsing training_label.txt
s=pd.read_csv(sys.argv[1],delimiter="\n",header=None)
# print(s)
s=np.array(s.iloc[:,[0]])

setence=np.empty(len(s), dtype=object)
label=np.zeros(len(s)).astype(str)

# print(s[5][0].split('+++$+++')[1])
for i in range(len(s)):
	setence[i]=s[i][0].split('+++$+++')[1]
	label[i]=s[i][0].split('+++$+++')[0]
# label=label.astype(int)
raw_setence=list(setence)
# print(setence)
print("raw_setemce",len(raw_setence))
label=list(label)
print(len(label))
print(type(label))
print(label[0])
for i in range(len(label)):
    label[i]=label[i].split()[0]

setence=setence.reshape(len(s),1)
# print(setence)
ssetence=[setence[i][0].split() for i in range(len(s))]
# for i in range(10):
	# print(ssetence[i])	
# print(len(ssetence))
# print(ssetence)

#parsing testing_data



all_data=S+ssetence


model = Word2Vec(all_data, min_count=10,sg=0,size=200,window=10)
#model=Word2Vec.load('newmodel.bin') 
#model.save('newmodel.bin')
#model = Word2Vec.load('newmodel.bin')
print(model)

# model.save('newmodel.bin')
tok = Tokenizer()
raw_data=raw_x+raw_setence
#print(raw_data[10000])
tok.fit_on_texts(raw_data)
# print(t.document_count)
#pk.dump(tok,open(os.getcwd()+'/tok.pk','wb'))
vocab_size = len(tok.word_index) + 1
#print(vocab_size)

encoded_docs = tok.texts_to_sequences(raw_setence)
padded_docs = pad_sequences(encoded_docs, maxlen=40, padding='post')

embed_matrix = np.zeros((vocab_size,200))
for word, i in tok.word_index.items():
	if word in model:
		embed_matrix[i] = model[word]
#print(embed_matrix)
# print(embed_matrix.shape)

checkpointer = keras.callbacks.ModelCheckpoint('great0523086.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
rnn_model = Sequential()
e = Embedding(vocab_size,200, weights=[embed_matrix], input_length=40, trainable=False)
rnn_model.add(e)

rnn_model.add(GRU(256, return_sequences=True,dropout=0.4,recurrent_dropout=0.4))
rnn_model.add(GRU(128, dropout=0.4,recurrent_dropout=0.4))
#rnn_model.add(LSTM(32, dropout=0.2))
rnn_model.add(Dense(32))
rnn_model.add(Dropout(0.4))
rnn_model.add(Activation('relu'))
#rnn_model.add(keras.layers.PReLU())
rnn_model.add(Dense(1))
rnn_model.add(Activation('sigmoid'))
#rnn_model.add(keras.layers.PReLU())
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(rnn_model.summary())
rnn_model.fit(padded_docs, label, epochs=100, batch_size=1024, callbacks=[checkpointer],validation_split=0.1)

rnn_model.save('great0523086.hdf5')

