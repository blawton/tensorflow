import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

ypred = tf.nn.softmax(tf.matmul(x, W) + b)

yreal = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(yreal * tf.log(ypred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for it in range(1000):
    xbatch, ybatch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:xbatch, yreal:ybatch})

labelpred = tf.equal(tf.argmax(yreal,1), tf.argmax(ypred,1))
accuracy = tf.reduce_mean(tf.cast(labelpred, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, yreal: mnist.test.labels}))

for i in range(10):
    weights = W.eval()
    W_i = weights[:,i]
    W_i = W_i.reshape((28,28))
    plt.imshow(W_i)
    plt.show()
