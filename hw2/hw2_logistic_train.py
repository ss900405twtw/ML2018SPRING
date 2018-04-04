import numpy as np 
import csv
from math import exp
import pandas as pd 
import sys
# train_X = open(sys.argv[1] ,"r")
# train_Y = open(sys.argv[2] ,"r")
# input(train_X)


def simulity(data1,data2):
	count=0
	for i in range(32561):
		if data1[i] == data2[i]:
			count=count+1
	return count/32561




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






def sigmoid(data):
	return np.tanh(data*0.5)/2+0.5


#normalization
data_x=pd.DataFrame(data_x)
normalized_df=(data_x-data_x.mean())/data_x.std()
data_x=normalized_df	
# data_x=data_x.iloc[:,[0,4,12,15,17,18,22,32,36,37,38,41,43,47,49,50,56,65,66,67,68,69,70,76,77,78,79,80]]
data_x=data_x.as_matrix()




data_x = np.concatenate((np.ones((data_x.shape[0],1)),data_x), axis=1)
data_x = np.concatenate(((data_x[:,1]**2).reshape(32561,1),data_x),axis=1)
data_x = np.concatenate(((data_x[:,[80,81,82]]**2).reshape(32561,3),data_x),axis=1)
data_x = np.concatenate(((data_x[:,[81,82]]**2).reshape(32561,2),data_x),axis=1)
# 增加bias項  


data_y=data_y.reshape(32561,1)

parlen=len(data_x[0])
ydim=len(data_x)
b=0.0
w=np.zeros(parlen).astype('float').reshape(parlen,1)
sum_b=0.0
sum_w=np.zeros(parlen).astype('float').reshape(parlen,1)
# w=np.matrix(w)
one=np.ones(ydim).astype('float').reshape(ydim,1)
one_b=np.ones(ydim).astype('float').reshape(ydim,1)
# w=np.matrix(w)
lr=0.00001
iteration=10000
b_grad=0.0
w_grad=np.zeros(parlen).astype('float').reshape(parlen,1)
sum_b=0
sum_w=np.zeros(parlen).astype('float').reshape(parlen,1)
Sum=0.0

yy=[]

data_y=np.array(data_y)
for i in range(len(data_y)):
	yy.append(data_y[i][0])
yy=np.array(yy)





for i in range(iteration):
    
    sigmoid_f=sigmoid(np.dot(data_x,w))  

    

    w_grad = - np.dot(data_x.T,(data_y-sigmoid_f))
    Sum = - np.dot(data_y.T,np.log(sigmoid_f+ 1e-20))-np.dot((1-data_y).T,np.log(1-sigmoid_f + 1e-20))
    print(i,Sum/ydim)
    

    # s=sigmoid_f.astype('float')
    # s=np.array(s) 
    # x=[]
    # for j in range(len(data_x)):
	   #  x.append(s[j][0])

    # x=np.array(x)
    # x=np.where(x>=0.5,1.0,x)
    # x=np.where(x<0.5,0.0,x)

    # print(i,simulity(x,yy))



   
    w=w-lr*w_grad

np.save('model.npy',w)


s=sigmoid(np.dot(data_x,w)).astype('float')
s=np.array(s)
x=[]
for i in range(len(data_x)):
	x.append(s[i][0])

x=np.array(x)
x=np.where(x>=0.5,1.0,x)
x=np.where(x<0.5,0.0,x)

yy=[]

data_y=np.array(data_y)
for i in range(len(data_y)):
	yy.append(data_y[i][0])
yy=np.array(yy)
# print(yy)



print(simulity(x,yy))



test_y=[]
with open(sys.argv[5],mode= 'r',encoding = "ISO-8859-1") as csvfile:
	reader=csv.reader(csvfile)
	for row in reader:
		test_y.append(row)
test_y.pop(0)
test_y=np.array(test_y).astype('float')

test_y=pd.DataFrame(test_y)
normalized=(test_y-test_y.mean())/test_y.std()
test_y=normalized
test_y.iloc[:,112]=0.0	
test_y=test_y.as_matrix()






# print(simulity(x,yy))


pm=[]

test_y = np.concatenate((np.ones((test_y.shape[0],1)),test_y), axis=1)
test_y = np.concatenate(((test_y[:,1]**2).reshape(16281,1),test_y),axis=1)
# test_y = np.concatenate(((test_y[:,80]**2).reshape(16281,1),test_y),axis=1)
test_y = np.concatenate(((test_y[:,[80,81,82]]**2).reshape(16281,3),test_y),axis=1)
# test_y = np.concatenate(((test_y[:,13]**2).reshape(16281,1),test_y),axis=1)
test_y = np.concatenate(((test_y[:,[81,82]]**2).reshape(16281,2),test_y),axis=1)
# 增加bias項  
sc=sigmoid(np.dot(test_y,w)).astype('float')
sc=np.array(sc)

test=[]
for i in range(len(test_y)):
	test.append(sc[i][0])

test=np.array(test)
test=np.where(test>=0.5,1,test)
test=np.where(test<0.5,0,test)


o=open(sys.argv[6],'a')
o.write("id")
o.write(",")
o.write("label")
o.write("\n")
for k in range(1,16282):
	# o.write("id_")
	o.write(str(k))
	o.write(",")
	o.write(str(int(test[k-1])))
	o.write("\n")
o.close()



# pm=[]

# test_y = np.concatenate((np.ones((test_y.shape[0],1)),test_y), axis=1)
# test_y = np.concatenate(((test_y[:,1]**2).reshape(16281,1),test_y),axis=1)
# # 增加bias項  
# sc=sigmoid(np.dot(test_y,w)).astype('float')
# sc=np.array(sc)

# test=[]
# for i in range(len(test_y)):
# 	test.append(sc[i][0])

# test=np.array(test)
# test=np.where(test>=0.5,1,test)
# test=np.where(test<0.5,0,test)


# o=open("plz2.csv",'a')
# o.write("id")
# o.write(",")
# o.write("label")
# o.write("\n")
# for k in range(1,16282):
# 	# o.write("id_")
# 	o.write(str(k))
# 	o.write(",")
# 	o.write(str(int(test[k-1])))
# 	o.write("\n")
# o.close()




# b=list(b)
# w=list(w)
