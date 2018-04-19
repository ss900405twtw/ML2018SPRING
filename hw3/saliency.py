import os
import argparse
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
import numpy
# from utils import *
# from termcolor import colored,cprint

x = pd.read_csv("train.csv")
x=x.values
X = np.zeros((len(x), 48*48))
Y = np.zeros((len(x), 1))
for i in range(len(x)):
    X[i,:]=x[i,1].split(' ')
    Y[i]=x[i,0]

avg=np.mean(X, axis=0)    
st=np.std(X, axis=0)
X=(X-avg)/st
x_train = X.astype('float')
# img_rows, img_cols = 48, 48
# batch_size = 100
# nb_classes = 7
# nb_epoch = 400

# Train_x=X[:25838,:]
# Val_x=X[25838:,:]

# Train_x = numpy.asarray(Train_x) 
# Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols,1)

# Val_x = numpy.asarray(Val_x)
# Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols,1)

# x_train = X.astype('float')
# Val_x = Val_x.astype('float32')


# # Train_x=X[:1,:][0]
# # Train_y=Y[:1,:]
# Val_y=Y[25838:,:]
# print(X.shape)


model_name = "Model69.hdf5"

emotion_classifier = load_model(model_name)
input_img = emotion_classifier.input

# x_train = load_data()

sess = K.get_session()
for idx in range(5):
    val_proba = emotion_classifier.predict(x_train[idx].reshape(1, 48, 48, 1))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classifier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])
    heatmap = None
    thres = 0.5
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    heatmap = grads.eval(session=sess, feed_dict={input_img:x_train[idx].reshape(1, 48, 48, 1)}).reshape(48,48)
    see = x_train[idx].reshape(48, 48)
    see[np.where(heatmap <= thres)] = np.mean(see)
    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('color'+str(idx)+'.jpg', dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('gray'+str(idx)+'.jpg', dpi=100)   