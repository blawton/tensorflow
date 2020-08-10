import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.get_variable("D_W1", shape=[784, 128],
           initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.get_variable("D_W2", shape=[128, 1],
           initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.get_variable("G_W1", shape=[100, 128],
           initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.get_variable("G_W2", shape=[128, 784],
           initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

#Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
with sess.as_default():   # or `with sess:` to close on exit
    for x in range(0, 30000):
        X_mb, _ = mnist.train.next_batch(5)
        if x % 2000 == 0:
            data_sample=sess.run(generator(sample_Z(1, 100)))
            image_sample = np.reshape(data_sample, (28, 28))*255
            plt.imshow(image_sample, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            plt.show()
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(5, 100)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(5, 100)})