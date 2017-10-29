"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import os
import matplotlib.pyplot as plt
import numpy
import random
import time
import glob
import tensorflow as tf
from mnistdata import MnistData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = time.time()

# Parameters
learning_rate = 0.0001
training_epochs = 10000
batch_size = 100
logs_path = 'logs'

# Set the logs folder and delete any files in it

r = glob.glob(logs_path + '/*')
for i in r:
    os.remove(i)

mnist = MnistData()

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
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

with tf.name_scope('Loss'):
    # Cost function definition
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

with tf.name_scope('Optimiser'):
    # Define the optimisation algorithm
    #optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    # Model evaluation
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
        
        batch_xs, batch_ys = mnist.train_random_batch(batch_size)

        _, l, summary = sess.run([optimiser, cost, merged_summary_op],
                                 feed_dict={x: batch_xs, y: batch_ys})

        summary_writer.add_summary(summary, i)

    finalaccuracy = sess.run(accuracy,
                             feed_dict={x: mnist.test_data, y: mnist.test_labels})
    print("Classification accuracy for the test data set: " + str(finalaccuracy))

end_time = time.time()
total_time = end_time - start_time
print("Runtime: " + str(total_time) + " seconds.")

# print("Enter to the command line the following: tensorboard --logdir=" + logs_path)
# print("Using an internet browser navigate to: http://localhost:6006")
