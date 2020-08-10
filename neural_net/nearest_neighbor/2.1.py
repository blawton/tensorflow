import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtrain, Ytrain = mnist.train.next_batch(2000)
Xtest, Ytest = mnist.test.next_batch(10000)

x = tf.placeholder("float", [2000, 784])
y = tf.placeholder("float", [2000, 10])
xt = tf.placeholder("float", [784])

distance = -tf.reduce_sum(tf.square(x - tf.matmul(tf.ones(x.get_shape()), tf.diag(xt))), axis=1)

values, indices = tf.nn.top_k(distance, k=5, sorted=True)

nn = tf.convert_to_tensor([y[index] for index in tf.unstack(indices)])

label = tf.argmax(tf.reduce_sum(nn, axis=0))

for j in range(5):
    if (label != tf.argmax(nn[j])):
        break
    j += 1

prediction = nn[j,:]
TestClass = np.ones((10000,10))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    acc = 0.0
    for h in range(len(Xtest)):
        nn_pred = sess.run(prediction, feed_dict={x: Xtrain, y: Ytrain, xt: Xtest[h,:]})
        TestClass[h,:] = nn_pred
        if np.all(nn_pred == Ytest[h,:]):
            acc += 1
    acc = acc/(len(Xtest))
    print(acc)
print(Xtest)
for i in range(10):
    TestClass_i= TestClass[:,i]
    matches_i = np.where(TestClass_i == 1)
    pixelmatch_i = Xtest[matches_i, :][0]
    img_i = np.sum(pixelmatch_i, axis=0) / len(pixelmatch_i)
    img_i = img_i.reshape((28, 28))
    plt.imshow(img_i)
    plt.show()
