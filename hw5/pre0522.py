import numpy as np
import pandas as pd
import re
import csv
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import _pickle as pk
from keras.layers import Dense, Dropout, Activation, Flatten
'''
x=pd.read_csv("training_nolabel.txt",delimiter="\n",header=None)
xx=np.array(x.iloc[:,[0]])
y=np.array(x.iloc[:,[0]]).reshape(len(x),)
# print(x)
raw_x=list(y)
# print(raw_x)
# print(x[0][0])
# print(re.findall(r"[\w']+", x[0][0]))
S=[xx[i][0].split() for i in range(len(x))]
# print(S)



# print(S)
# for i in range(10):
    # print(S[i])   



f = open('training_nolabel.txt','r')  
result = list()  

for line in open('training_nolabel.txt'):  
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
s=pd.read_csv("training_label.txt",delimiter="\n",header=None)
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
'''
#parsing testing_data
num=pd.read_csv(sys.argv[1],delimiter="\n",header=None)
# print(len(num))
# print(num)
with open(sys.argv[1]) as csvfile:
    reader=csv.reader(csvfile)
    reader=list(reader)
    w=np.empty(len(num), dtype=object)
    test_setence=np.empty(len(num), dtype=object)
    test_label=np.zeros(len(num)).astype(str)
    count_line=0
    for row in reader:      
        w[count_line]=row
        test_setence[count_line]=','.join(row[1:])
        test_label[count_line]=row[0][0]
        count_line=count_line+1
# print(test_setence[1])
test_label=test_label[1:]

raw_testsettence=list(test_setence)
#print("len of raw setence",len(raw_setence))

raw_testsettence=raw_testsettence[1:]

test_setence=test_setence[1:].reshape(-1,1)
test_label=test_label.astype(int)
ttest_setence=[test_setence[i][0].split() for i in range(len(num)-1)]
# print(type(ttest_setence[0]))
# print(type(ttest_setence[0]))
# print(np.append(ttest_setence,S,axis=0))

# for i in range(len(S)+len(ssetence)+len(ttest_setence)):
#all_data=S+ssetence+ttest_setence



max_length=40
#raw_data=raw_x+raw_setence+raw_testsettence

if(sys.argv[3]=='public'):
	modelrnn = keras.models.load_model('great0523084.hdf5')
else:
	modelrnn = keras.models.load_model('great0523086.hdf5')
tok=pk.load(open('tok.pk','rb'))
vocab_size = len(tok.word_index)+1

encoded_docs_test = tok.texts_to_sequences(raw_testsettence)
max_length=40
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
'''
t = Tokenizer()
t.fit_on_texts(raw_data)
vocab_size = len(t.word_index) + 1
#print(encoded_docs)


encoded_docs_test = t.texts_to_sequences(raw_testsettence)
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
'''
#print(padded_docs_test)

result=modelrnn.predict(padded_docs_test,batch_size=1024,verbose=1)

#model30=keras.models.load_model('rnnModeldroplet_30.hdf5')
#result30=model30.predict(padded_docs_test)

#d=(result+result30)/2
d=result
f = open(sys.argv[2],"w")
w = csv.writer(f)
k=0
w.writerow(('id','label'))

for i in d:
    w.writerow((str(k),int(np.round(i))))
    k=k+1
f.close()

