import numpy as np 
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


#parsing movie.csv


'''

not use now

'''

#parsing test.csv

'''

not use now

'''

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
print("user len: ",user_len)
print("movie len: ",movie_len)
model = get_model(user_len,movie_len)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print("movie_emb shape: ",movie_emb.shape)
#model.summary()

#input(user_train)
#input(movie_train)
#input(rating_data)
checkpointer = keras.callbacks.ModelCheckpoint('movie.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model.fit([user_data,movie_data], rating_data, epochs=47, batch_size=4096, callbacks=[checkpointer],validation_data=([user_test,movie_test],rating_test))

#a.fit()`
