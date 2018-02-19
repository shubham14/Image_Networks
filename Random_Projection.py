 #--------------- Random Projection implementation for Neural Networks ---------------#

import numpy as np
import pandas as pd
#import tensorflow as tf
from cifar_parser import *
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, initializers
from keras.regularizers import l2
import timeit
from sklearn.preprocessing import *
import matplotlib.pyplot as plt

start = timeit.default_timer()

# data is being loaded by unpickling the data
# train-test split is being kept as 0.2 or 20% 
X_train, y_train, X_test, y_test = data_unpickle(0.2)

# data shape charecterstics
m = int(X_train.shape[0])
# d = int(X_train.shape[1])
d = 3072
# sequential model with random projection  
model = Sequential()

# Network parameters
k = 1000
n = 500
# number of labels
l = 10

y_train = keras.utils.to_categorical(y_train, l)
y_test = keras.utils.to_categorical(y_test, l)

# lists to store accuracy scores
Random_Projection_scores = []
Vanilla_scores = []
x = []
c = 1;
for i in range(int(k/3),k,200):

	# model which has IID gaussian layer of dimensions m * m/2
	model.add(Dense(i, kernel_initializer="random_normal",input_dim = d, trainable= False))
	model.add(Dense(k, input_dim = i, activation='relu'))

	# add another layer
	model.add(Dense(i, kernel_initializer="random_normal",input_dim = k, trainable= False))
	model.add(Dense(k, input_dim = i, activation='relu'))

	# add another layer
	model.add(Dense(i, kernel_initializer="random_normal",input_dim = k, trainable= False))
	model.add(Dense(n, input_dim = i, activation='relu'))

	# output layer
	model.add(Dense(i, kernel_initializer="random_normal",input_dim = n, trainable= False))
	model.add(Dense(l, input_dim = i, activation='relu'))

	model.compile(loss='categorical_crossentropy',
	          optimizer='sgd',
	          metrics=['accuracy'])

	# Fit model
	model.fit(X_train, y_train, batch_size=32, nb_epoch=10)

	# Evaluate model on test data
	score = model.evaluate(X_test, y_test)

	#------------------ Regular Dense Neural Network -----------------------

	# model without random projection
	model1 = Sequential()

	# model which has IID gaussian layer of dimensions m * m/2
	model1.add(Dense(k, input_shape = (d,), activation = 'relu'))

	# add another layer
	model1.add(Dense(k, activation = 'relu'))

	# add another layer
	model1.add(Dense(n, activation = 'relu'))

	# output layer
	model1.add(Dense(l, activation = 'relu'))

	model1.compile(loss='categorical_crossentropy',
	          optimizer='sgd',
	          metrics=['accuracy'])

	# Fit model
	model1.fit(X_train, y_train, batch_size = 32, epochs = 10)

	# Evaluate model on test data
	score1 = model1.evaluate(X_test, y_test)

	x.append(c)
	c = c + 1

# print accuracy metrics
print ("The accuracy for the network without Random Projection layer :")
print (score1)
print ("The accuracy for the network with Random Projection layer :")
print (score)

plt.title("Random Projection Neural Network vs Vanilla Neural Network")
plt.plot(Random_Projection_scores, x, 'r-', Vanilla_scores, x, 'g-')
stop = timeit.default_timer()

# for timing the program run time
print ("Total elapsed time")
print (stop - start)

# saving the plot
savefig('Rate.png')
