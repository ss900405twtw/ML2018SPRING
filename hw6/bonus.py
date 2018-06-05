import numpy as np
import sys
import csv 
import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import *
#from keras.optimizer import *
from keras import optimizers
from keras.optimizers import Adam,Nadam,Adamax
import keras

def get_model(n_users,n_items,latent_dim=666):
	#ADAM = Adam(lr=0.01)
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec= Embedding(n_users,latent_dim,embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec= Embedding(n_users,latent_dim,embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(n_users,1,embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_users,1,embeddings_initializer='zeros')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1,normalize=True)([user_vec,item_vec])
	r_hat = Add()([r_hat,user_bias,item_bias])
	model = keras.models.Model([user_input,item_input],r_hat)
	model.compile(loss='mse',optimizer=Adam())
	return model


#parsing user.csv
f = open('users.txt','r',encoding ="ISO-8859-1")
result = list()
category=list()
gend=dict()
boys=dict()
girls=dict()
for line in open('users.txt',encoding="ISO-8859-1"):
    line = f.readline()
    s=line.split('::')
    gend[s[0]]=s[1]
'''
	if s[1]=='M':
        boys[s[0]]=s[1]
    elif s[1]=='F':
        girls[s[0]]=s[1]
'''
#print(boys)
#print('boys length: ',len(boys))
#print('girls length: ',len(girls))
#print(boys[1])






'''



#parsing test.csv

'''

#not use now



#parsing train.csv ---- user + rating

tra = pd.read_csv('train.csv')
user_data = np.array(tra['UserID'])



# user_input = np.array(tra['UserID'])
user_len=len(np.unique(user_data))
user_num = len(user_data)
user_train=user_data[:round(user_num*0.9)]
user_test=user_data[round(user_num*0.9):]

movie_data = np.array(tra['MovieID'])
# item_input = np.array(tra['MovieID'])
movie_len=len(np.unique(movie_data))
movie_num = len(movie_data)
movie_train=movie_data[:round(movie_num*0.9)]
movie_test=movie_data[round(movie_num*0.9):]



rating_data = np.array(tra['Rating'] ).astype('float')
rating_train = rating_data[:round(movie_num*0.9)]
rating_test = rating_data[round(movie_num*0.9):]



gender=np.empty(user_num,dtype='object')
for i in range(user_num):
	#print(gend['796'])
	#print(type(user_data[i].astype('str')))
	gender[i]=gend[user_data[i].astype('str')]
#print(gender)
#print(len(gender))
tra['gender']=gender
#print(tra)

#classify girls and boys

#user_boys_data=np.zeros(800000)
#user_girls_data=np.zeros(800000)
#movie_boys_data=np.zeros(800000)
#movie_girks_data=np.zeros(800000)
#rating_boys_data=np.zeros(800000)
#rating_girs_data=np.zeros(800000)
want_boys = tra.loc[tra['gender']=='M',['UserID','MovieID','Rating']]
want_girls = tra.loc[tra['gender']=='F',['UserID','MovieID','Rating']]
#print(want_girls)
#print(len(want_girls))

user_boys_data=np.array(want_boys['UserID'])
user_girls_data=np.array(want_girls['UserID'])
movie_boys_data=np.array(want_boys['MovieID'])
movie_girls_data=np.array(want_girls['MovieID'])
rating_boys_data=np.array(want_boys['Rating'])
rating_girls_data=np.array(want_girls['Rating'])






#print("user len: ",user_len)
#print("movie len: ",movie_len)
model = get_model(user_len,movie_len)
#movie_emb = np.array(model.layers[3].get_weights()).squeeze()
#print("movie_emb shape: ",movie_emb.shape)
model.summary()
'''
#input(user_train)
#input(movie_train)
#input(rating_data)

checkpointer = keras.callbacks.ModelCheckpoint('girls.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
model.fit([user_girls_data,movie_girls_data], rating_girls_data, epochs=110, batch_size=4096, callbacks=[checkpointer])
'''
#a.fit()`

girls_model=keras.models.load_model('girls.hdf5')
boys_model=keras.models.load_model('boys.hdf5')

test = pd.read_csv('test.csv')
user_test_data=np.array(test['UserID'])
movie_test_data=np.array(test['MovieID'])
test_gender=np.empty(len(user_test_data),dtype='object')
for i in range(len(user_test_data)):
	#print(user_test_data[i],type(user_test_data[i]))
	test_gender[i]=gend[user_test_data[i].astype('str')]
#print(test_gender)
#print(len(test_gender))

test['gender']=test_gender	
gender_test=np.array(test['gender'])
#print(test)
#ans=np.zeros(len(user_data),dtype='float')
#for i in range(len(user_test_data)):
#	if gender_test[i]=='M':
#		print(boys_model.predict(['3203','2126'],batch_size=4096,verbose=1))
	#elif gender_test[i]=='F':
	#	ans[i]=girls_model.predict([user_test_data[i],movie_test_data[i]],batch_size=4096,verbose=1)

index=0
boys_index=[]
girls_index=[]
print("start indexing")
for i in range(len(gender_test)):
	if gender_test[i]=='M':
		boys_index.append(i)
	elif gender_test[i]=='F':
		girls_index.append(i)

#print(girls_index)
#print(len(girls_index))
#print(len(boys_index))


boy_dataframe=test.loc[test['gender']=='M',['UserID','MovieID','Rating']]
girl_dataframe=test.loc[test['gender']=='F',['UserID','MovieID','Rating']]

boytestuser=np.array(boy_dataframe['UserID'])
boytestmovie=np.array(boy_dataframe['MovieID'])
#boytestrating=np.array(boy_dataframe['Rating'])


girltestuser=np.array(girl_dataframe['UserID'])
girltestmovie=np.array(girl_dataframe['MovieID'])
#girltestrating=np.array(girl_dataframe['Rating'])

boy_answer=boys_model.predict([boytestuser,boytestmovie],batch_size=4069,verbose=1)
girl_answer=girls_model.predict([girltestuser,girltestmovie],batch_size=4096,verbose=1)
#print(girl_answer[-1][0])
#print(len(boy_answer))
total_answer=np.concatenate((boy_answer,girl_answer),axis=0)
total_answer=total_answer.reshape(1,-1)[0]
total_index=np.array(boys_index+girls_index)
print(boys_index)
#print(len(boys_index))
ans_dict=dict()
for i in range(len(total_index)):
	ans_dict[total_index[i].astype('str')]=total_answer[i]
print("whe length of dict:",len(ans_dict))


ans=np.zeros(len(ans_dict),dtype='float')
#ans=np.zeros(len(total_answer))
for i in range(len(total_answer)):
	ans[i]=ans_dict[str(i)]


print(ans)
print(len(ans))
'''
'''
f = open(sys.argv[1],"w")
w = csv.writer(f)
w.writerow(('TestDataID','Rating'))

for i in range(len(ans)):    
    if(ans[i]>5):
        ans[i]=5.0
    if(ans[i]<1):
        ans[i]=1.0
    w.writerow((str(i+1),float(ans[i])))
f.close()  	



