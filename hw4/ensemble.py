import numpy as np 
import pandas as pd 
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers import *
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import csv


import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
x = pd.read_csv(sys.argv[1])
x=x.values
X = np.zeros((len(x), 48*48))
Y = np.zeros((len(x), 1))
for i in range(len(x)):
    X[i,:]=x[i,1].split(' ')
    Y[i]=x[i,0]

avg=np.mean(X, axis=0)    
st=np.std(X, axis=0)
X=(X-avg)/st

img_rows, img_cols = 48, 48
batch_size = 100
nb_classes = 7
nb_epoch = 50

Train_x=X[:25838,:]
Val_x=X[25838:,:]

Train_x = numpy.asarray(Train_x) 
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols,1)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')

Train_y=Y[:25838,:]
Val_y=Y[25838:,:]
drop_rate=0.2
Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)
datagen = ImageDataGenerator(
     featurewise_center=False,
     samplewise_center=False,
     featurewise_std_normalization=False,
     samplewise_std_normalization=False,
     zca_whitening=False,
     rotation_range=10,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True,
     vertical_flip=False)

model = Sequential()
model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(256, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(256, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(256, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(Convolution2D(512, (3, 3), padding='same'))
model.add(keras.layers.PReLU())
model.add(BatchNormalization())
#model.add(Dropout(drop_rate))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(keras.layers.PReLU())
model.add(Dense(64))
model.add(keras.layers.PReLU())
#model.add(Dense(64))
#model.add(keras.layers.PReLU())
model.add(Dense(7, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


filepath='Model69.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='max')
#early_stop =keras.callbacks.EarlyStopping(monitor='val_acc', patience=1, mode='max')
#checkpointer = [checkpoint, early_stop]


datagen.fit(Train_x)
model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    #steps_per_epoch=Train_x.shape[0],
                    epochs=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[checkpointer])

model = keras.models.load_model("Model.hdf5")


x = pd.read_csv(sys.argv[1])
x=x.values
X = np.zeros((len(x), 48*48))
Y = np.zeros((len(x), 1))
for i in range(len(x)):
    X[i,:]=x[i,1].split(' ')
    Y[i]=x[i,0]

avg=np.mean(X, axis=0)    

st=np.std(X, axis=0)
X=(X-avg)/st

XT = numpy.asarray(X) 
img_rows, img_cols = 48, 48
XT = XT.reshape(XT.shape[0],img_rows,img_cols,1)
result=model.predict(XT)

k=0
f = open(sys.argv[2],"w")
w = csv.writer(f)
k=0
w.writerow(('id','label'))

for i in result:
    w.writerow((str(k),np.argmax(i)))
    k=k+1
f.close()
'''                    

def maxRepeating(arr, n,  k):
 
    # Iterate though input array, for every element
    # arr[i], increment arr[arr[i]%k] by k
    for i in range(0,  n):
        arr[arr[i]%k] += k
 
    # Find index of the maximum repeating element
    max = arr[0]
    result = 0
    for i in range(1, n):
     
        if arr[i] > max:
            max = arr[i]
            result = i
 
    # Uncomment this code to get the original array back
    #for i in range(0, n):
    #    arr[i] = arr[i]%k
 
    # Return index of the maximum element
    return result
# xgboost=pd.read_csv('xgboost.csv')
treee=pd.read_csv('output.csv')
adaa=pd.read_csv('outputplz.csv')
knnn=pd.read_csv('outputokok.csv')
# rff=pd.read_csv('rf.csv')
# lgg=pd.read_csv('lg.csv')


# xgboost=np.array(xgboost['label']).astype('float')
treee=np.array(treee['label']).astype('int')
adaa=np.array(adaa['label']).astype('int')
knnn=np.array(knnn['label']).astype('int')
treeee=list(treee)
adaa=list(adaa)
knnn=list(knnn)

# lgg=np.array(lgg['label']).astype('float')
# rff=np.array(rff['label']).astype('float')
'''
pred=[]
for i in range(len(tree)):
	a=list([knn[i],treee[i],adaa[i]])
	# b=list([knnn[i],lgg[i],knnn[i],lgg[i]])
	if((a.count(0)+b.count(0))>(a.count(1)+b.count(1))):
		pred.append(0)
	else:
		pred.append(1)
# print(pred)
'''
pred=[]
for i in range(len(treee)):
	a=list([knnn[i],treee[i],adaa[i]])
	# b=list([knnn[i],lgg[i],knnn[i],lgg[i]])
	if(len(set(a))==3):
		pred.append(knnn[i])
	elif(knnn[i]==treee[i]):
		pred.append(knnn[i])
	elif(knnn[i]==adaa[i]):
		pred.append(knnn[i])	
	elif(treee[i]==adaa[i]):
		pred.append(treee[i])
	# else:
		# pred.append(knnn[i])






o=open("vote2.csv",'a')
o.write("id")
o.write(",")
o.write("label")
o.write("\n")
for i in range(0,len(treee)):
    o.write(str(i))
    o.write(",")
    o.write(str(int(pred[i])))
    o.write("\n")
o.close()





