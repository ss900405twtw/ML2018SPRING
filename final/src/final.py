import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
from opencc import OpenCC
#tradition to simple
opencc=OpenCC('hk2s')

#parsing data
s1=pd.read_csv("1_train.txt", names = ["Sequence"])
s2=pd.read_csv("2_train.txt", names = ["Sequence"])
s3=pd.read_csv("3_train.txt", names = ["Sequence"])
s4=pd.read_csv("4_train.txt", names = ["Sequence"])
f = open('5_train.txt','r')  
result = list()  
for line in open('5_train.txt'):  
    line = f.readline()  
    result.append(line)  
set5=[i.split("\n")[0] for i in result]

set1=np.array(s1['Sequence'])
set2=np.array(s2['Sequence'])
set3=np.array(s3['Sequence'])
set4=np.array(s4['Sequence'])
set5=np.array(set5)
#print('start')
#input(s3)

split_len=4

splt1=[]
#jieba1=[]
print(len(set1)//split_len)
for i in range(len(set1)-split_len):
	splt1.append(set1[i]+","+set1[i+1]+","+set1[i+2]+","+set1[i+3])
#splt1[-1]=splt1[-1][:-1]
#input(splt1)
print(len(splt1))




jieba.set_dictionary('dict.txt.big')
splt1=[opencc.convert(splt1[i]) for i in range(len(splt1))]
jieba1=[jieba.cut(splt1[i],cut_all=False) for i in range(len(splt1))]
jieba_1=[]
for i in range(len(jieba1)):
	raw=[]
	for word in jieba1[i]:
		raw.append(word)
	jieba_1.append(raw)

#input(jieba_1)


#print(jieba_1)
#print(splt1)



splt2=[]
print(len(set2)//split_len)
for i in range(len(set2)-split_len):
	splt2.append(set2[i]+","+set2[i+1]+","+set2[i+2]+","+set2[i+3])
#splt1[[1]=splt1[-1][:-1]
#print(splt2)
print(len(splt2))

jieba.set_dictionary('dict.txt.big')
splt2=[opencc.convert(splt2[i]) for i in range(len(splt2))]

jieba2=[jieba.cut(splt2[i],cut_all=False) for i in range(len(splt2))]
jieba_2=[]
for i in range(len(jieba2)):
	raw=[]
	for word in jieba2[i]:
		raw.append(word)
	jieba_2.append(raw)
#print(jieba_1+jieba_2)
#print(len(jieba_1+jieba_2))

splt3=[]
#print(len(set3)//split_len)
for i in range(len(set3)-split_len):
	splt3.append(set3[i]+","+set3[i+1]+","+set3[i+2]+","+set3[i+3])
#splt1[[1]=splt1[-1][:-1]
#print(splt3)
print(len(splt3))

jieba.set_dictionary('dict.txt.big')
splt3=[opencc.convert(splt3[i]) for i in range(len(splt3))]

jieba3=[jieba.cut(splt3[i],cut_all=False) for i in range(len(splt3))]
jieba_3=[]
for i in range(len(jieba3)):
	raw=[]
	for word in jieba3[i]:
		raw.append(word)
	jieba_3.append(raw)
#print(jieba_3)


splt4=[]
#print(len(set4)//split_len)
for i in range(len(set4)-split_len):
	splt4.append(set4[i]+","+set4[i+1]+","+set4[i+2]+","+set4[i+3])
#splt1[[1]=splt1[-1][:-1]
#splt4.append(set4[len(set4)//split_len*4]+set4[len(set4)//split_len*4+1]+set4[len(set4)//split_len*4+2])

print(len(splt4))

jieba.set_dictionary('dict.txt.big')
splt4=[opencc.convert(splt4[i]) for i in range(len(splt4))]

jieba4=[jieba.cut(splt4[i],cut_all=False) for i in range(len(splt4))]
jieba_4=[]
for i in range(len(jieba4)):
	raw=[]
	for word in jieba4[i]:
		raw.append(word)
	jieba_4.append(raw)
#print(jieba_4)


splt5=[]
print(len(set5)//split_len)
for i in range(len(set5)-split_len):
	splt5.append(set5[i]+","+set5[i+1]+","+set5[i+2]+","+set5[i+3])
#print(splt5)
#print(len(splt5))
#splt1[[1]=splt1[-1][:-1]

jieba.set_dictionary('dict.txt.big')
splt5=[opencc.convert(splt5[i]) for i in range(len(splt5))]

jieba5=[jieba.cut(splt5[i],cut_all=False) for i in range(len(splt5))]
jieba_5=[]
for i in range(len(jieba5)):
	raw=[]
	for word in jieba5[i]:
		raw.append(word)
	jieba_5.append(raw)

#print(jieba_5)

#parsing testing data
s=pd.read_csv('testing_data.csv')
#print(s.iloc[:,2])
s1=np.array(s.iloc[:,1])
s1_split=[s1[i].replace("\t", "") for i in range(len(s1))]
s1_sp=[s1_split[i].replace(" ", "") for i in range(len(s1))]
#print(s1_sp[-2][-6])
s1_sp=[s1_sp[i].replace(s1_sp[-2][-6], "") for i in range(len(s1))]
#print(s1_sp)
jieba.set_dictionary('dict.txt.big')
s1_sp=[opencc.convert(s1_sp[i]) for i in range(len(s1_sp))]

jieba6=[jieba.cut(s1_sp[i],cut_all=False) for i in range(len(s1_sp))]
jieba_6=[]
for i in range(len(jieba6)):
    raw=[]
    for word in jieba6[i]:
        raw.append(word)
    jieba_6.append(raw)
#print(jieba_6)

s2=np.array(s.iloc[:,2])
s2_split=[s2[i].replace("\t", "") for i in range(len(s2))]
s2_sp=[s2_split[i].replace(" ", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("1", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("2", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("3", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("4", "") for i in range(len(s2))]
s2_sp=[s2_sp[i].replace("5", "") for i in range(len(s2))]
s2_spl=[s2_sp[i].split(':')[1:] for i in range(len(s2))]
#print(s2_spl)

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
#print(jieba_7[9])



















print("jieba finished")
total_jieba=jieba_1+jieba_2+jieba_3+jieba_4+jieba_5+jieba_6+jieba_7
#print(total_jieba)
print(len(total_jieba))
model = Word2Vec(total_jieba,min_count=1,sg=1,size=38,window=20)
model.save('0624_49_window38.bin')

print(model)





