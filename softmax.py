import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random
import time

start_time = time.time()

# def trainevalmodel(training_epochs, learning_rate, batch_size):
# Parameters
learning_rate = 1.5
training_epochs = 2000
batch_size = 100
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
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Weights variable definition and initialisation
W = tf.Variable(tf.zeros([784, 10]), name='Weights')

# Bias variable definition and initialisation
b = tf.Variable(tf.zeros([10]), name='Bias')

with tf.name_scope('Model'):
    # Model definition
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

with tf.name_scope('Loss'):
    # Cost function definition
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

with tf.name_scope('Optimiser'):
    # Define the optimisation algorithm
    optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    # Model evaluation
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialise the variables
init = tf.global_variables_initializer()

# Add summary data to monitor the optimisation of the model
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()
print("Summary information defined.")

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    # Train the model
    for i in range(training_epochs):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      _, l, summary = sess.run([optimiser, cost, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
      #_, l = sess.run([optimiser, cost], feed_dict={x: batch_xs, y: batch_ys})

      summary_writer.add_summary(summary, i)

    finalaccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Classification accuracy for the test data set: " + str(finalaccuracy))

end_time = time.time()
total_time = end_time - start_time
print("Runtime: " + str(total_time) + " seconds.")

print("Ended to the command line the following command:/n tensorboard --logdir=logs")
print("Using an internet browser navigate to http://localhost:6006")
