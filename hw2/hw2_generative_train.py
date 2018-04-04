import numpy as np 
import csv
from math import exp
import pandas as pd 
import sys


def simulity(data1,data2):
	count=0
	for i in range(32561):
		if data1[i] == data2[i]:
			count=count+1
	return count/32561

def sigmoid(data):
	return np.tanh(data*0.5)/2+0.5


def predict(X_test,mu1,mu2,shared_sigma,N1,N2):
	sigma_inverse=np.linalg.pinv(shared_sigma)
	w=np.dot((mu1-mu2),sigma_inverse)
	x=X_test.T 
	b=(-0.5)*np.dot(np.dot([mu1],sigma_inverse),mu1) \
		+(0.5)*np.dot(np.dot([mu2],sigma_inverse),mu2)+np.log(float(N1)/N2)
	a=np.dot(w,x)+b
	y=sigmoid(a)
	for i in range(len(X_test)):
		if y[i]<0.5:
			y[i]=0
		elif y[i]>=0.5:
			y[i]=1



	return y


train=[]
test_raw=[]
with open(sys.argv[3],mode= 'r',encoding = "ISO-8859-1") as csvfile:
	reader=csv.reader(csvfile)
	for row in reader:
		train.append(row)
train.pop(0)
# print(len(train))
#32561

with open(sys.argv[4],mode= 'r',encoding = "ISO-8859-1") as csvfile:
	reader=csv.reader(csvfile)
	for row in reader:
		test_raw.append(row)
# print(test_raw[1])
test=[]
for i in range(len(test_raw)):
	test.append(test_raw[i][0])
# print(len(test))
data_x=np.array(train).astype('float')
data_y=np.array(test).astype('float')

test_y=[]
with open(sys.argv[5],mode= 'r',encoding = "ISO-8859-1") as csvfile:
	reader=csv.reader(csvfile)
	for row in reader:
		test_y.append(row)
test_y.pop(0)
test_y=np.array(test_y).astype('float')


data_x=pd.DataFrame(data_x)
normalized_df=(data_x-data_x.mean())/data_x.std()
data_x=normalized_df	


data_x=data_x.as_matrix()

test_y=pd.DataFrame(test_y)
# print(test_y.iloc[:,112])
normalized=(test_y-test_y.mean())/test_y.std()





test_y=normalized
test_y.iloc[:,112]=0.0	


test_y=test_y.as_matrix()

X_train=data_x
Y_train=data_y


#mean
dim=len(data_x[0])
train_data_size=32561
cnt1=0
cnt2=0
mu1=np.zeros((dim,))
mu2=np.zeros((dim,))
for i in range(train_data_size):
	if Y_train[i]==1:
		mu1+=X_train[i]
		cnt1+=1
	else:
		mu2+=X_train[i]
		cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

#sigma
sigma1=np.zeros((dim,dim))
sigma2=np.zeros((dim,dim))
for i in range(train_data_size):
	if Y_train[i]==1:
		sigma1 += np.dot(np.transpose([X_train[i]-mu1]),[(X_train[i]-mu1)])
	else:
		sigma2 += np.dot(np.transpose([X_train[i]-mu2]),[(X_train[i]-mu2)])
sigma1 /= cnt1
sigma2 /= cnt2

shared_sigma=(float(cnt1) / train_data_size)*sigma1 \
				+(float(cnt2) / train_data_size)*sigma2


pry=predict(test_y,mu1,mu2,shared_sigma,cnt1,cnt2)
# print(cnt1)
# print(cnt2)
o=open(sys.argv[6],'a')
o.write("id")
o.write(",")
o.write("label")
o.write("\n")
for k in range(1,16282):
	# o.write("id_")
	o.write(str(k))
	o.write(",")
	o.write(str(int(pry[k-1])))
	o.write("\n")
o.close()
