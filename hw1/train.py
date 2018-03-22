#more data go go go !!!
import pandas as pd
import numpy as np
import sys
import csv 
import math
import random


train=pd.read_csv("train.csv",encoding="ISO-8859-1")



aa=pd.DataFrame({})
for k in range(0,20*18,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    aa=pd.concat([aa,s],axis=1)



# s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
# 		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]


bb=pd.DataFrame({})
for k in range(20*18,20*18*2,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    bb=pd.concat([bb,s],axis=1)


cc=pd.DataFrame({})
for k in range(20*18*2,20*18*3,18):
    # s=train.iloc[0+i:18+i,3:]
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    cc=pd.concat([cc,s],axis=1)

dd=pd.DataFrame({})
for k in range(20*18*3,20*18*4,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    dd=pd.concat([dd,s],axis=1)
ee=pd.DataFrame({})
for k in range(20*18*4,20*18*5,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    ee=pd.concat([ee,s],axis=1)
ff=pd.DataFrame({})
for k in range(20*18*5,20*18*6,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    ff=pd.concat([ff,s],axis=1)
gg=pd.DataFrame({})
for k in range(20*18*6,20*18*7,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    gg=pd.concat([gg,s],axis=1)


hh=pd.DataFrame({})
for k in range(20*18*7,20*18*8,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    hh=pd.concat([hh,s],axis=1)


ii=pd.DataFrame({})
for k in range(20*18*8,20*18*9,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    ii=pd.concat([ii,s],axis=1)

jj=pd.DataFrame({})
for k in range(20*18*9,20*18*10,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    jj=pd.concat([jj,s],axis=1)
kk=pd.DataFrame({})
for k in range(20*18*10,20*18*11,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    kk=pd.concat([kk,s],axis=1)
ll=pd.DataFrame({})
for k in range(20*18*11,20*18*12,18):
    s=train.iloc[[0+k,1+k,2+k,3+k,4+k,5+k,6+k,7+k,
		8+k,9+k,11+k,12+k,13+k,14+k,15+k,16+k,17+k],3:]
    s=s.reset_index(drop=True)
    ll=pd.concat([ll,s],axis=1)
    

plz=pd.concat([aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll])

# print(plz)
ww=np.array([])
for k in range(0,202,17):
	for i in range(0,471):
		s=plz.iloc[[8+k],0+i:9+i]		
		s=s.values
		# print(s)
		ww=np.append(ww,s)

ww=ww.astype('float')  

for i in range(len(ww)):
	if ww[i]<=0 or ww[i]>=300:
		ww[i]=ww[i-1]
data_x=ww.reshape(5652,9)
# print(data_x[0])

ww=np.array([])
for k in range(0,202,17):
	for i in range(0,471):
		s=plz.iloc[[9+k],0+i:9+i]		
		s=s.values
		# print(s)
		ww=np.append(ww,s)

ww=ww.astype('float')  

for i in range(len(ww)):
	if ww[i]<=0 or ww[i]>=300:
		ww[i]=ww[i-1]
data_n=ww.reshape(5652,9)

old=np.array([])
new=np.array([])
for i in range(5652):
	old=np.append(data_x[i],data_n[i])
	# old=np.append(old,data_z[i])
	new=np.append(new,old)
new=new.reshape(5652,9*2)
data_x=new





data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()
x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)
data_y=y
for i in range(len(data_y)):
	if data_y[i]<=0 or data_y[i]>300:
		data_y[i]=data_y[i-1]




b=0.0
w=np.zeros(18).astype('float')
lr=1
iteration=30000
sum_b=0
sum_w=np.zeros(18).astype('float')
for i in range(iteration):
    b_grad=0.0
    w_grad=np.zeros(18).astype('float')
    sum=0
    for n in range(len(data_x)):
        b_grad = b_grad - 2.0*(data_y[n]-b-(np.dot(w,(data_x[n].reshape(18,1)))))*1.0

        w_grad = w_grad - 2.0*(data_y[n]-b-(np.dot(w,(data_x[n].reshape(18,1)))))*data_x[n]
        sum=sum+(data_y[n]-b-(np.dot(w,(data_x[n].reshape(18,1)))))**2
    print(i,(1/5652*sum)**0.5)
    sum_b=sum_b+b_grad**2
    sum_w=sum_w+w_grad**2
    b=b-lr / ((sum_b)**0.5) * b_grad
    w=w-lr / ((sum_w)**0.5) * w_grad


b=list(b)
w=list(w)

with open("ww.txt",'a') as d:
	d.write(str(w))
d.close()

with open("bb.txt",'a') as c:
	c.write(str(b))
c.close()

