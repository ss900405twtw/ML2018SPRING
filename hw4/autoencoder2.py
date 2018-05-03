import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import keras
#from keras.layers import *
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition





'''

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
		'''


encoder = keras.models.load_model("encoder.h5")
#encoded_imgs = encoder.predict(X)
#encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
#kmeans = KMeans(n_clusters = 2,random_state=0).fit(encoded_imgs)

Y = np.load('visualization.npy')
Y = Y.astype('float32' ) / 255.
Y = np.reshape(Y,(len(Y),-1))
encoded_imgs = encoder.predict(Y)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
print(encoded_imgs.shape)
kmeans = KMeans(n_clusters = 2,random_state=0).fit(encoded_imgs)
labels=kmeans.labels_
print(encoded_imgs.shape)
comp=encoded_imgs
print(list(labels).count(0))



p=[]
for i in range(10000):
     p.append(labels[i])
x_embedded = TSNE(n_components=2).fit_transform(comp)

print("x_embedded:",x_embedded.shape)
'''
plt.scatter(x_embedded[:5000, 0],x_embedded[:5000,1],c='b',label='dataset A',s=0.2)
plt.scatter(x_embedded[5000:, 0],x_embedded[5000:,1],c='r',label='dataset B',s=0.2)
plt.legend()
plt.savefig('tsne.png')

'''
p1=np.array([])
p2=np.array([])
for j in range(len(p)):
	if p[j]==0:
		p1=np.concatenate((p1,x_embedded[j]),axis=0)

	else:
		p2=np.concatenate((p2,x_embedded[j]),axis=0)

print("p1:",len(p1))
print("p2:",len(p2))
#print(p1[0])
#p1=np.array([p1])
#p2=np.array([p2])
p1=p1.reshape(len(p1)//2,2)
p2=p2.reshape(len(p2)//2,2)

print("p1 shape: ",p1.shape)
print("p2 shape: ",p2.shape)

	
plt.scatter(p1[:len(p2), 0],p1[:len(p2),1],c='b',label='dataset A',s=0.2)
plt.scatter(p2[:len(p1), 0],p2[:len(p1),1],c='r',label='dataset B',s=0.2)
plt.legend()
plt.savefig('raw.png')



'''
f = pd.read_csv('test_case.csv')
IDs, idx1, idx2 = np.array(f['ID']),np.array(f['image1_index']),np.array(f['image2_index'])

o=open('autoencoder2.csv','w')
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
'''


#encoded_imgs = encoder.predict(X)
#encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
#kmeans = KMeans(n_clusters = 2,random_state=0).fit(encoded_imgs)



