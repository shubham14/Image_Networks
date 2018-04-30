# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 11:31:10 2018

@author: Shubham
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, learning_rate, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
    conv1 = layers.conv2d(inputs = input_layer, filters = 32, 
                          kernel_size = [5, 5],
                          padding = "same",
                          activation = tf.nn.relu)
    
    pool1 = layers.max_pooling2d(inputs = conv1, pool_size = [2, 2],
                                    strides = 2)
    
    conv2 = layers.conv2d(inputs = pool1, filters = 64, 
                          kernel_size = [5, 5],
                          padding = "same",
                          activation = tf.nn.relu)
    pool2 = layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
    
    pool2_flat = tf.reshape(pool2, [-1 , 7* 7* 64])
    dense = layers.dense(inputs = pool2_flat, pool_size = [2, 2],
                         activation = tf.nn.relu)
    dropout = layers.dropout(inputs = dense,
                             rate = 0.4,
                             training = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = layers.dense(inputs = dropout, units = 10)
    
    predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        # could add custom optimizers here
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss = loss,
                                      global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, 
                                          loss = loss,
                                          train_op = train_op)
    eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(
                    labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
    
def main():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
