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
import numpy
import pandas as pd

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


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
