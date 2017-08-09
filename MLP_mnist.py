#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:43:15 2017

@author: kathy yu
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist= input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

#weights initialization 
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros(10))

#feed the x with input dataset 
x = tf.placeholder(tf.float32,[None,in_units])
#use as parms of dropout
keep_prob = tf.placeholder(tf.float32)
#relu
hidden1 = tf.nn.relu(tf.matmul(x,W1)+ b1)
#dropout
hidden2 = tf.nn.dropout(hidden1,keep_prob)
#output with softmax
y = tf.nn.softmax(tf.matmul(hidden2, W2)+ b2)

#define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(0.3).minimize(cross_entropy)

#train
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y: batch_ys, keep_prob: 0.75})
    
#test trained accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (accuracy.eval({x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))




