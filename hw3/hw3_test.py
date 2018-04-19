import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import csv
import numpy
import pandas as pd

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
if sys.argv[3]=='public':
	model = keras.models.load_model("Model69.hdf5")
elif sys.argv[3]=='private':
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
