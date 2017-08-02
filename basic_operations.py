import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random
import time
import glob, os

# Set the logs folder and delete any files in it
logs_path = 'logs'
r = glob.glob(logs_path + '/*')
for i in r:
   os.remove(i)

################################################################################
## Graph nodes creation
print("\n###### Graph nodes creation ######")

# Create two constant graph nodes that represent constant values.
print("\nCreate two constant graph nodes that represent constant values")
node_con_1 = tf.constant(1.0, dtype=tf.float32)
node_con_2 = tf.constant(2.0) # also tf.float32 implicitly
print(node_con_1, node_con_2)

# Create a graph node that adds the values that represent two other nodes.
node_add_1 = tf.add(node_con_1, node_con_2)
node_add_2 = node_con_1 + node_con_2
print(node_add_1, node_add_2)

# Create two graph nodes that represent tf.variables
node_var_1 = tf.Variable(1, dtype=tf.float32)
node_var_2 = tf.Variable(2, dtype=tf.float32)

################################################################################
## Graph and node evaluation ways
print("\n###### Graph and node evaluation ways ######")

# Tensor evaluation ways
print("\nTensorFlow session ")

# Initialise a TensorFlow session
with tf.Session() as sess:

    # Set current tf.Session as the default one
    sess.as_default()
    # Get the default session
    sess = tf.get_default_session()

    # Initialise  all the variables
    sess.run(tf.global_variables_initializer())

    # Tensor evaluation ways
    print("\nTensor evaluation ways")
    print("print(node_con_1.eval()) : " + str(node_con_1.eval()))
    result = sess.run(node_con_1)
    print("print(sess.run(node_con_1)) : " + str(result))
    result = sess.run(node_add_1)
    print("print(sess.run(node_add_1)) : " + str(result))
    result = sess.run(node_add_2)
    print("print(sess.run(node_add_2)) : " + str(result))

    # Change tf.Variable values
    print("\n###### Change tf.Variable values ######")
    print("print(node_var_1.eval()) : " + str(node_var_1.eval()))
    sess.run(node_var_1.assign(2))
    print("print(node_var_1.eval()) : " + str(node_var_1.eval()))

    # Define the node for the addition of two tf.variables
    node_add_3 = node_var_1 + node_var_1
    print("node_add_3: " + str(node_add_3))

    node_var1_assign = node_var_1.assign(node_add_3)

    for i in range(5):
        # The code below evaluates the graph that adds node_var_1 to node_var_2
        # and stores the result in node_var_1
        print(sess.run(node_var1_assign))
