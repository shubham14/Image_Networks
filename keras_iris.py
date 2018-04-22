# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:20:48 2018

@author: Shubham
"""

#required library which holes the iris dataset
from sklearn.datasets import load_iris
#One Hot Encode our Y:
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential #Sequential Models
from keras.layers import Dense #Dense Fully Connected Layer Type
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer
from Keras_custom_layer import *
import keras
import time
import matplotlib.pyplot as plt

#load the iris dataset
iris = load_iris()
#our inputs will contain 4 features
X = iris.data[:, 0:4]
#the labels are the following
y = iris.target
#print the distinct y labels
print(np.unique(y))

encoder = LabelBinarizer()
Y = encoder.fit_transform(y)

fraction = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# parameter lists contains duration, accuracy and loss for plotting
acc = []
duration = []
loss = []

# keras claaback for storing losses
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def baseline_model_mask(ele):
    model = Sequential()
    model.add(MyLayer(input_shape = (4,), output_dim = 4, p = ele, init='normal',  activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


for ele in fraction:
    start = time.time()
    neural_network = baseline_model_mask(ele)
    history = LossHistory()
    neural_network.fit(X,Y, epochs=500, batch_size=10)
    end = time.time()
    a, l = neural_network.evaluate(X,Y)
    acc.append(l)
    loss.append(a)
    duration.append(end-start)
    
    
plt.savefig('acc1.png')
plt.plot(fraction, acc)

plt.savefig('duration.png')
plt.plot(fraction, duration)

plt.savefig('acc-duration.png')
plt.plot(duration,acc)

#
#[20.455251455307007,
# 20.02211284637451,
# 20.980010509490967,
# 23.045305013656616,
# 25.18450951576233,
# 24.154337882995605,
# 21.062286853790283,
# 24.452695846557617,
# 24.182719945907593]
#
#[0.9533333349227905,
# 0.9600000039736429,
# 0.9533333349227905,
# 0.9600000039736429,
# 0.9533333349227905,
# 0.88,
# 0.7666666666666667,
# 0.6666666666666666,
# 0.6666666666666666]
