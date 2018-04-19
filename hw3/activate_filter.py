from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
# from marcos import *
import numpy as np
import numpy
import pandas as pd
import PIL
from PIL import Image
import matplotlib.pyplot as plt






fig = plt.figure(figsize=(14, 8))

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
# print(Train_x[0].shape)
Train_x = numpy.asarray(Train_x)
Train_x = Train_x.reshape(Train_x.shape[0],48,48,1)
# Train_x[0]=np.array(Train_x[0]
# print(Train_x[:1,:].shape)
Train_x=Train_x[:1,:]




# model = applications.VGG16(include_top=False,
#                            weights='imagenet')
model=load_model("Model69.hdf5")
# print(model)

# get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])


# emotion_classifier = load_model("Model69.hdf5")
  
# emotion_classifier = load_model(model)


layer_dict = dict([(layer.name, layer) for layer in model.layers])
# layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

input_img = model.input

# layer_name = 'block5_conv3'

layer_name = 'conv2d_2'
num_filter=64
for j in range(num_filte):
	filter_index = j  # can be any integer from 0 to 511, as there are 512 filters in that layer

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	loss = K.mean(layer_output[:, :, :, filter_index])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	input_img_data = Train_x
	# print(input_img_data.shape)
	# run gradient ascent for 20 steps
	for i in range(20):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * 20
	def deprocess_image(x):
		# normalize tensor: center on 0., ensure std is 0.1
		x -= x.mean()
		x /= (x.std() + 1e-5)
		x *= 0.1

		# clip to [0, 1]
		x += 0.5
		x = np.clip(x, 0, 1)

		# convert to RGB array
		x *= 255
		x = x.transpose((1, 2, 0))
		x = np.clip(x, 0, 255).astype('uint8')
		return x

	# img = input_img_data[0]
	img = input_img_data[0]

	img = deprocess_image(img)
	img=img.reshape(48,48)
	# print(img.shape)
	ax = fig.add_subplot(num_filte/16, 16, j+1)
	ax.imshow(img, cmap='BuGn')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()
fig.suptitle('Filters of layer')
img_path=os.getcwd()
fig.savefig(os.path.join(img_path,'filter'))

	# img=img.reshape(48,48)
	# img = Image.fromarray(img.reshape(48,48))
	# img.show()
	# imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
