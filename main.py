### Workshop 18th July 2018

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)							# the same initial weights

from keras.models import Sequential 					# sequential ?? 
from keras.layers import Dense, Dropout, Activation, Flatten		# setting NN architecture

from keras.layers import Convolution2D, MaxPooling2D			# CNN

from keras.utils import np_utils

# MNIST
# Training 60000
# Testing  10000
# size 28*28 (gray image)
############################################### Clean Data ###########################################################################
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()		# load data for training and testing *setting proxy to download data

print(X_train.shape)

plt.imshow(X_train[0])
#plt.show()  								# show training data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)			# 1 --> one channel
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)			# (..., pixel, pixel, depth) --> TensorFlow

X_train = X_train.astype('float32')					# change integer to float --> more accuracy
X_test = X_test.astype('float32')

# Normalization
X_train/= 255								# X_train = X_train/255
X_test/= 255

#
Y_train = np_utils.to_categorical(y_train)				# y_train[2] = 4 and Y_train[2] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Y_test = np_utils.to_categorical(y_test)
############################################### Model ##############################################################################

model1 = Sequential()

model1.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Convolution2D(32, (5, 5), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Flatten())							# from 2d-image to vector
model1.add(Dense(128, activation='relu'))
model1.add(Dense(10, activation='softmax'))

# model1.summary() --> show our achitecture

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=6, batch_size=32, verbose=2, shuffle=True)	
# dropout --> protect overfitting
# epochs --> number of repeat
# shuffle = True
############################################# 
Y_predict = model1.predict_classes(X_test)
for i in range(9):
	plt.subplot(3, 3, i+1)
	idx = np.random.randint(10000)
	plt.imshow(X_test[idx].reshape([28, 28]))
	plt.title('Predict: '+ str(Y_predict[idx]) + str(Y_test[idx]))

plt.show()











