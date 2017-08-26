# @authors: Eleftherios Avramidis
# @email: el.avramidis@gmail.com
# @date: 2017/08/26
# @copyright: MIT License

import matplotlib.pyplot as plt
import numpy
import random
from tensorflow.examples.tutorials.mnist import input_data


class mnistdata(object):

    mnist = []            # Variable with the MNIST data

    traindata = []
    validationdata = []
    testdata = []

    ## Default constructor.
	#
	#  Sets the initial values of the object.
	#
	#  @param self                      is the reference to the current object.
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    def augmentdata(self):
        pass

if __name__ == "__main__":

    dataset = mnistdata()

    dataset.augmentdata()
