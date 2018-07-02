from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import jieba
from scipy import spatial
import sys
from opencc import OpenCC
opencc=OpenCC('hk2s')


model=Word2Vec.load('model.bin')
model2=Word2Vec.load('0624_49.bin')

print(model)
print(model2)

summ=0
for word ,vocab in model.wv.vocab.items():
	summ+=vocab.count
summ2=0
for word ,vocab in model2.wv.vocab.items():
	summ2+=vocab.count

print("total length is ",summ)
print("total length of 2 is ",summ2)



vocab_len=len(model.wv.vocab)
vocab2_len=len(model2.wv.vocab)

print(vocab_len)
print(vocab2_len)
#input()
model=model2
print("current model: ")
print(model)
#input()
#print(model2.wv.vocab[","].count/summ2)
#noneed=model2.wv.vocab[","].count
#input()


#formula:a/a+freq(obj)
def weighted(vocab):
	freq=model2.wv.vocab[vocab].count/(summ2)
	a=0.0025
	weight=a/(a+freq)
	#weight=1/((freq)**0.5)
	return np.array(model2[vocab])*weight
	

 




#print(model['拾'])
s=pd.read_csv('./dataset/testing_data.csv')
#print(s.iloc[:,2])
s1=np.array(s.iloc[:,1])
s1_split=[s1[i].replace("\t", "") for i in range(len(s1))]
s1_sp=[s1_split[i].replace(" ", "") for i in range(len(s1))]
#print(s1_sp[-2][-6])
s1_sp=[s1_sp[i].replace(s1_sp[-2][-6], "") for i in range(len(s1))]
jieba.set_dictionary('dict.txt.big')
s1_sp=[opencc.convert(s1_sp[i]) for i in range(len(s1_sp))]

jieba6=[jieba.cut(s1_sp[i],cut_all=False) for i in range(len(s1_sp))]
jieba_6=[]
for i in range(len(jieba6)):
    raw=[]
    for word in jieba6[i]:
        raw.append(word)
    jieba_6.append(raw)
#print(s1_sp)
question=[]  
for i in range(len(jieba_6)):
    vec=np.zeros(45)
    for j in range(len(jieba_6[i])):
        vec=vec+weighted(jieba_6[i][j])
    #print(type(vec))
    #print(len(jieba_7[i]))
    question.append(vec/len(jieba_6[i])) 
#print(question[1])
#print(len(question))







s2=np.array(s.iloc[:,2])
s2_split=[s2[i].replace("\t", "") for i in range(len(s2))]
s2_sp=[s2_split[i].replace(" ", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("1", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("2", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("3", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("4", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("5", "") for i in range(len(s2))]
s2_spl=[s2_sp[i].split(':')[1:] for i in range(len(s2))]
#print(s2_spl[1])

#result=[]
#s2=[ result.extend(el) for el in s2_spl]

s2=[j for i in s2_spl for j in i]
jieba.set_dictionary('dict.txt.big')
s2=[opencc.convert(s2[i]) for i in range(len(s2))]

jieba7=[jieba.cut(s2[i],cut_all=False) for i in range(len(s2))]
jieba_7=[]
for i in range(len(jieba7)):
    raw=[]
    for word in jieba7[i]:
        raw.append(word)
    jieba_7.append(raw)


#print(jieba_7)

#for i in range(len(s2)):
#	for item in s2_spl:
opt=[]	
for i in range(len(jieba_7)):
	vec=np.zeros(45)
	for j in range(len(jieba_7[i])):
		vec=vec+weighted(jieba_7[i][j])
	#print(type(vec))
	#print(len(jieba_7[i]))
	opt.append(vec/len(jieba_7[i]))	
#print(opt[1])
#print(len(opt))
#print(len(opt))

#print(s2_spl)

print(1-spatial.distance.cosine(question[0], opt[0]))

ans=[]
for i in range(5060):
	result=np.zeros(6)
	for j in range(6):
		result[j]=1-spatial.distance.cosine(question[i], opt[i*6+j])	
		#print(result[j])

	ans.append(np.argmax(result))
#ans = np.dtype(ans.int32)
#ans=ans.astype(np.int32)

#result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
print(ans[:30])
print(len(ans))
#ans=int(ans)
o=open('0624_49.csv','a')
o.write("id")
o.write(",")
o.write("ans")
o.write("\n")
for k in range(5060):
    #o.write("id_")
    o.write(str(k))
    o.write(",")
    o.write(str(ans[k]))
    o.write("\n")
o.close()



















#print(model['我'])



