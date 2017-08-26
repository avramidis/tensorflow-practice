#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random
import time
from scipy.ndimage import rotate
from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
from skimage.transform import radon, rescale

start_time = time.time()

# Parameters
learning_rate = 1e-4
training_epochs = 20000
batch_size = 100
display_step = 1000
logs_path = 'logs'

# Set the logs folder and delete any files in it
import glob, os
r = glob.glob(logs_path + '/*')
for i in r:
   os.remove(i)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input variable definition and initialisation
x = tf.placeholder(tf.float32, [None, 784], name='InputData')

# Definition of the variable for the target values
y_ = tf.placeholder(tf.float32, [None, 10], name='LabelData')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='Weights')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='Bias')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('Model'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 128])
    b_conv1 = bias_variable([128])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 128, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Dropout
    # h_fc0_drop = tf.nn.dropout(h_pool2, keep_prob)

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    # Dropout connect
    keep_prob = tf.placeholder(tf.float32)
    W_fc1_drop = tf.nn.dropout(W_fc1, keep_prob)

    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1_drop) + b_fc1)

    # Dropout
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

with tf.name_scope('Loss'):
    # Cost function definition
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

with tf.name_scope('Optimiser'):
    # Model optimisation
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)


with tf.name_scope('Accuracy'):
    # Model evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    validation_accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add summary data to monitor the optimisation of the model
cost_sum = tf.summary.scalar("cost", cross_entropy)
accuracy_sum = tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge([cost_sum, accuracy_sum])
#merged_summary_op = tf.summary.merge_all()

validation_accuracy_sum = tf.summary.scalar("validation_accuracy", validation_accuracy2)
merged_summary_val = tf.summary.merge([validation_accuracy_sum])

print("Summary information defined.")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    for i in range(training_epochs):
        if i % display_step == 0:
            validation_accuracy = accuracy.eval(feed_dict={
                            x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            print('step %d, validation accuracy %g' % (i, validation_accuracy))

            _, summary = sess.run([validation_accuracy2, merged_summary_val],
                                    feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})

            summary_writer.add_summary(summary, i)

        batch = mnist.train.next_batch(batch_size)

        # for b in range(batch_size):
        #     # r=rotate(numpy.reshape(batch[0][b], (28, 28)), random.randint(-15,15), reshape=False)
        #     # batch[0][b]=numpy.reshape(r, (784))
        #     # image = r
        #
        #     # thresh = threshold_mean(batch[0][b])
        #     # batch[0][b] = batch[0][b] > thresh
        #
        #     # r=numpy.reshape(batch[0][b], (28, 28))
        #     # r = skeletonize(r)
        #     # batch[0][b]=numpy.reshape(r, (784))
        #
        #     r=numpy.reshape(batch[0][b], (28, 28))
        #     theta = numpy.linspace(0., 180., max(r.shape), endpoint=False)
        #     r = radon(r, theta=theta, circle=True)
        #     batch[0][b]=numpy.reshape(r, (784))

        _, l, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

        summary_writer.add_summary(summary, i)

    print('validation accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))

    # for t in mnist.test.images:
        # thresh = threshold_mean(t)
        # t = t > thresh
        # r=numpy.reshape(t, (28, 28))
        # r = skeletonize(r)
        # t=numpy.reshape(r, (784))
        # r=numpy.reshape(batch[0][b], (28, 28))
        # theta = numpy.linspace(0., 180., max(r.shape), endpoint=False)
        # r = radon(r, theta=theta, circle=True)
        # batch[0][b]=numpy.reshape(r, (784))

    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

end_time = time.time()
total_time = end_time - start_time
print("Runtime: " + str(total_time) + " seconds.")

print("Enter to the command line the following: tensorboard --logdir=" + logs_path)
print("Using an internet browser navigate to: http://localhost:6006")
