import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

#from keras.layers import *
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import pandas as pd 







train_num =130000
X = np.load('image.npy')
X = X.astype('float32' ) / 255.
X = np.reshape(X,(len(X),-1))
x_train = X[:train_num]
x_val =X[train_num:]
print(x_train.shape,x_val.shape)




input_img = Input(shape=(784,))

encoded = Dense(128,activation='relu')(input_img)

#encoded = Dense(128,activation='relu')(encoded)
#encoded = Dense(256,activation='Prelu')(encoded)

encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(32,activation='relu')(encoded)

decoded = Dense(64,activation ='relu')(encoded)
decoded = Dense(128,activation ='relu')(decoded)

decoded = Dense(256,activation ='relu')(decoded)

#decoded = Dense(512,activation ='relu')(decoded)
decoded = Dense(784,activation ='relu')(decoded)
encoder = Model(input = input_img,output=encoded)

adam = Adam(lr=5e-4)
autoencoder = Model(input = input_img,output = decoded)
autoencoder.compile(optimizer = adam ,loss ='mse')
autoencoder.summary()

autoencoder.fit(x_train,x_train,
				epochs=50,
				batch_size=256,
				shuffle=True,
				validation_data=(x_val,x_val))
autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')

encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
kmeans = KMeans(n_clusters = 2,random_state=0).fit(encoded_imgs)


f = pd.read_csv('test_case.csv')
IDs, idx1, idx2 = np.array(f['ID']),np.array(f['image1_index']),np.array(f['image2_index'])

o=open('autoencoder.csv','w')
o.write("ID,Ans\n")
for idx ,i1,i2 in zip(IDs, idx1, idx2):
	p1 = kmeans.labels_[i1]
	p2 = kmeans.labels_[i2]
	if p1 == p2:
		pred = 1
	else:
		pred =0
	o.write("{},{}\n".format(idx, pred))
o.close()



