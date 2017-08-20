import matplotlib.pyplot as plt
import numpy
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Print four random samples from the training images
r=random.randint(0,len(mnist.train.images))
f, axarr = plt.subplots(2, 2, sharex=False)
axarr[0][0].imshow(numpy.reshape(mnist.train.images[r+0], (28, 28)))
axarr[0][0].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
axarr[0][1].imshow(numpy.reshape(mnist.train.images[r+1], (28, 28)))
axarr[0][1].set_title(numpy.where(mnist.train.labels[r+1]==1)[0][0])
axarr[1][0].imshow(numpy.reshape(mnist.train.images[r+2], (28, 28)))
axarr[1][0].set_title(numpy.where(mnist.train.labels[r+2]==1)[0][0])
axarr[1][1].imshow(numpy.reshape(mnist.train.images[r+3], (28, 28)))
axarr[1][1].set_title(numpy.where(mnist.train.labels[r+3]==1)[0][0])
f.subplots_adjust(hspace=0.3)
plt.show(block=False)

# Rotate samples
from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt

f, axarr = plt.subplots(2, 2, sharex=False)
img=rotate(numpy.reshape(mnist.train.images[r+0], (28, 28)), 30, reshape=False)
axarr[0][0].imshow(img)
axarr[0][0].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
img=rotate(numpy.reshape(mnist.train.images[r+1], (28, 28)), 30, reshape=False)
axarr[0][1].imshow(img)
axarr[0][1].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
img=rotate(numpy.reshape(mnist.train.images[r+2], (28, 28)), 30, reshape=False)
axarr[1][0].imshow(img)
axarr[1][0].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
img=rotate(numpy.reshape(mnist.train.images[r+3], (28, 28)), 30, reshape=False)
axarr[1][1].imshow(img)
axarr[1][1].set_title(numpy.where(mnist.train.labels[r+0]==1)[0][0])
f.subplots_adjust(hspace=0.3)
plt.show(block=True)
