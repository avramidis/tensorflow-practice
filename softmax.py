import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random

# Set the logs folder and delete any files in it
import glob, os
logspath = 'logs/*'
r = glob.glob(logspath)
for i in r:
   os.remove(i)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Print four random samples from the training images
# r=random.randint(0,len(mnist.train.images))
# f, axarr = plt.subplots(2, 2, sharex=False)
# axarr[0][0].imshow(numpy.reshape(mnist.train.images[r+0], (28, 28)))
# axarr[0][0].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
# axarr[0][1].imshow(numpy.reshape(mnist.train.images[r+1], (28, 28)))
# axarr[0][1].set_title(numpy.where(mnist.train.labels[r+1]==1)[0][0])
# axarr[1][0].imshow(numpy.reshape(mnist.train.images[r+2], (28, 28)))
# axarr[1][0].set_title(numpy.where(mnist.train.labels[r+2]==1)[0][0])
# axarr[1][1].imshow(numpy.reshape(mnist.train.images[r+3], (28, 28)))
# axarr[1][1].set_title(numpy.where(mnist.train.labels[r+3]==1)[0][0])
# f.subplots_adjust(hspace=0.3)
# plt.show(block=True)

# Input variable definition and initialisation
x = tf.placeholder(tf.float32, [None, 784])

# Weights variable definition and initialisation
W = tf.Variable(tf.zeros([784, 10]))

# Bias variable definition and initialisation
b = tf.Variable(tf.zeros([10]))

# Model definition
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cost function definition
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Define the optimisation algorithm
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Define and initilise a TensorFlow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
merged = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('logs', sess.graph)
tf.summary.scalar("cross_entropy", cross_entropy)
merged_summary_op = tf.summary.merge_all()
print("Summary information defined.")

# Train the model
for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, l, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict={x: batch_xs, y_: batch_ys})
  summary_writer.add_summary(summary, i)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Classification accuracy for the test data set: " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
