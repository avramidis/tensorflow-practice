import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import random
import time

# Set the logs folder and delete any files in it
import glob, os
r = glob.glob(logs_path + '/*')
for i in r:
   os.remove(i)

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
