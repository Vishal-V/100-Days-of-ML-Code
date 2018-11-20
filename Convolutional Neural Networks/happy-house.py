import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
from keras import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

def model(input_shape):
	#giving the image shape as param to the tensor
	X_input = Input(input_shape)

	#Layer 1 with pad->conv->batchnorm->activation
	X = ZeroPadding2D((3,3))(X_input)
	X = Conv2D(32, (7,7), name='conv0')(X)
	X = BatchNormalization(axis=3, name='bn0')(X)
	X = Activation('relu')(X)

	#Pooling layer
	X = MaxPooling2D((2,2), name='pool')(X)

	#flatten and fully connected layer
	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fc')(X)

	model = Model(inputs=X_input, outputs=X, name='model')

	return model

# Call model and initilaize it
model1 = model((128,128,3))

# Compile the model
model1.compile(optimizer='adam', loss='binary_crossentropy', megtrics=['accuracy'])

# train the model on the train set
model1.fit(x=X_train, y=Y_train, epochs=40, batch_size=16)

# test the model
preds = model1.evaluate(x=X_test, y=Y_test)

# print the accuracy
print(f'The loss is {preds[0]}')
print(f'The accuracy is {preds[1]}')



