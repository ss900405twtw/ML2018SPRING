
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
# from marcos import *
import numpy as np
import numpy
import pandas as pd





x = pd.read_csv("train.csv")
x=x.values
X = np.zeros((len(x), 48*48))
Y = np.zeros((len(x), 1))
# print(len(x))
for i in range(len(x)):
    X[i,:]=x[i,1].split(' ')
    Y[i]=x[i,0]

# avg=np.mean(X, axis=0)    
# st=np.std(X, axis=0)
# X=(X-avg)/st

img_rows, img_cols = 48, 48
batch_size = 100
nb_classes = 7
nb_epoch = 400

Train_x=X
Val_x=X

Train_x = numpy.asarray(Train_x) 
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols,1)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')

def main():
    emotion_classifier = load_model("Model69.hdf5")
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['conv2d_2','p_re_lu_1']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    private_pixels = Train_x
    # private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       # for i in range(len(private_pixels)) ]
    choose_id = 0
    photo = private_pixels[choose_id].reshape(1,48,48,1)
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os  .getcwd()
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

if __name__ == "__main__":
    main()