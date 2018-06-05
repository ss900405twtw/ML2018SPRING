import numpy as np 
import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import *
#from keras.optimizer import *
from keras import optimizers
from keras.optimizers import Adam,Nadam,Adamax
import keras
from sklearn import decomposition




def get_model(n_users,n_items,latent_dim=666):
	#ADAM = Adam(lr=0.01)
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec= Embedding(n_users,latent_dim,embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec= Embedding(n_items,latent_dim,embeddings_initializer='random_normal')(item_input)
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
#draw
'''
def draw(x,y):
	from matplotlib import pyplot as plt
	from tsne import bh_sne
	y=np.array(y)
	x=np.array(x,dtype=np.float64)
	vis_data=bh_sne(x)
	vis_x=vis_data[:,0]
	vis_y=vis_data[:,1]
	cm = plt.cm.get_cmap('RdYlBu')
	sc=plt.scatter(vis_x,vis_y,c=y,cmap=cm)
	plt.colorbar(sc)
	plt.show()
#sklearn version of tsne
'''
def draw(x,y):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    y=np.array(y)
    x=np.array(x,dtype=np.float64)
    vis_data=TSNE(n_components=2).fit_transform(x)
    vis_x=vis_data[:,0]
    vis_y=vis_data[:,1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc=plt.scatter(vis_x,vis_y,c=y,cmap=cm)
    plt.colorbar(sc)
    plt.show()

tra = pd.read_csv('train.csv')
user_data = np.array(tra['UserID'])



# user_input = np.array(tra['UserID'])
user_len=len(np.unique(user_data))
user_num = len(user_data)
user_train=user_data[:round(user_num*0.9)]
user_test=user_data[round(user_num*0.9):]

movie_data = np.array(tra['MovieID'])
# item_input = np.array(tra['MovieID'])
uniq_movie=np.unique(movie_data)
movie_len=len(np.unique(movie_data))
movie_num = len(movie_data)
print('unique movie is: ',type(uniq_movie))
print('unique movie len is :',len(uniq_movie))
movie_train=movie_data[:round(movie_num*0.9)]
movie_test=movie_data[round(movie_num*0.9):]

rating_data = np.array(tra['Rating'] ).astype('float')
rating_train = rating_data[:round(movie_num*0.9)]
rating_test = rating_data[round(movie_num*0.9):]
'''
model = get_model(user_len,movie_len)
#model.summary()
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print("user_emb shape: ",user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print("movie_emb shape: ",movie_emb.shape)
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)
print("save it")
'''
f = open('movies.txt','r',encoding ="ISO-8859-1")  
result = list()  
category=list()
uniq_movie=uniq_movie.astype('str')
for line in open('movies.txt',encoding="ISO-8859-1"):
    line = f.readline()
    s=line.split('::')
    #print(type(np.int64(s[0])))
    if s[0] in uniq_movie:
	    category.append(s[2])
	    result.append(line)



#category=category[1:]
print(len(category))
print("what is the fucking problem")
f.close()
for i in range(len(category)):
	category[i]=category[i].split('|')
	category[i][-1]=category[i][-1].split('\n')[0]
print(category[0])
#category=category[1:]
movie_category=[]
for i in range(len(category)):
	for item in category[i]:
		if item not in movie_category:
			movie_category.append(item)
print(category[2])
print("the total length is: ",len(category))

result.append(line)
class1=['Thriller','Horror','Crime','Mystery']
class2=['War','Adventure','Action','Western']
class3=['Drama','Musical','Romance','Documentary']
class4=['Sci-Fi','Fantasy','Film-Noir']
class5=["Children's",'Comedy','Animation']

for i in range(len(category)):
	for j in range(len(category[i])):
		if  category[i][j] in class1:
			category[i][j]=1
		elif category[i][j] in class2:
			category[i][j]=2
		elif category[i][j] in class3:
			category[i][j]=3
		elif category[i][j] in class4:
			category[i][j]=4
		elif category[i][j] in class5:
			category[i][j]=5
print(category[2])
movie_class=np.zeros(len(category))
for i in range(len(category)):
	if len(np.unique(category[i]))==len(category[i]):
		movie_class[i]=category[i][0]
	else:	
		movie_class[i]=np.bincount(category[i]).argmax()
	
print(movie_class)


#print(exist_movie)
#s=pd.read_csv('movies.csv',encoding='UTF-8') 
#print(s)
#parsing movie.csv


'''

not use now

'''

#parsing test.csv

'''

not use now

'''

#parsing train.csv ---- user + rating
'''
tra = pd.read_csv('train.csv')
user_data = np.array(tra['UserID'])



# user_input = np.array(tra['UserID'])
user_len=len(np.unique(user_data))
user_num = len(user_data)
user_train=user_data[:round(user_num*0.9)]
user_test=user_data[round(user_num*0.9):]

movie_data = np.array(tra['MovieID'])
# item_input = np.array(tra['MovieID'])
uniq_movie=np.unique(movie_data)
movie_len=len(np.unique(movie_data))
movie_num = len(movie_data)
print('unique movie is: ',uniq_movie)
print('unique movie len is :',len(uniq_movie))
movie_train=movie_data[:round(movie_num*0.9)]
movie_test=movie_data[round(movie_num*0.9):]

rating_data = np.array(tra['Rating'] ).astype('float')
rating_train = rating_data[:round(movie_num*0.9)]
rating_test = rating_data[round(movie_num*0.9):]

model = get_model(user_len,movie_len)
model.summary()
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print("user_emb shape: ",user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print("movie_emb shape: ",movie_emb.shape)
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)
print("save it")




draw(user_emb,movie_emb)
#input(user_train)
#input(movie_train)
#input(rating_data)

checkpointer = keras.callbacks.ModelCheckpoint('great0523086.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model.fit([user_data,movie_data], rating_data, epochs=46, batch_size=4096, callbacks=[checkpointer],validation_data=([user_test,movie_test],rating_test))

'''
model= keras.models.load_model('great0523086.hdf5')
model.summary()
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print("user_emb shape: ",user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print("movie_emb shape: ",movie_emb.shape)
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)
print("save it")

img = movie_emb.astype('float32')
pca=decomposition.PCA(n_components=50,whiten=True,svd_solver="full").fit(img.T)
comp = pca.components_
comp=comp.T
print(comp.shape)



draw(comp,movie_class)
#a.fit()
