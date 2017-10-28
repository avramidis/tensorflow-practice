# @authors: Eleftherios Avramidis
# @email: el.avramidis@gmail.com
# @date: 2017/08/26
# @copyright: MIT License

import numpy
import random
import copy
from tensorflow.examples.tutorials.mnist import input_data

from skimage.filters import threshold_mean
from skimage.morphology import skeletonize

from scipy.ndimage import rotate

class mnistdata(object):

    mnist = []            # Variable with the MNIST data

    trainimages = []

    validationimages = []

    testimages = []

    augmentedimages = []
    augmentedlabels = []

    ## Default constructor.
	#
	#  Sets the initial values of the object.
	#
	#  @param self                      is the reference to the current object.
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # self.mnist.test.images
        # self.mnist.test.labels
        #
        # self.mnist.train.images
        # self.mnist.train.labels

        self.augmentedimages = numpy.empty([len(self.mnist.train.images)*1, 784])
        self.augmentedlabels = numpy.empty([len(self.mnist.train.images)*1, 10])

    def datachange(self):
        for i in range(len(self.mnist.validation.images)):
            #image = copy.deepcopy(numpy.reshape(self.mnist.validation.images[i,:], (28, 28)))
            thresh = threshold_mean(self.mnist.validation.images[i,:])
            self.mnist.validation.images[i,:] = self.mnist.validation.images[i,:] > thresh

            # # perform skeletonization
            # image = skeletonize(image)

            #self.mnist.validation.images[i,:] = numpy.reshape(image, 784)

        for i in range(len(self.mnist.test.images)):
            #image = copy.deepcopy(numpy.reshape(self.mnist.test.images[i,:], (28, 28)))
            thresh = threshold_mean(self.mnist.test.images[i,:])
            self.mnist.test.images[i,:] = self.mnist.test.images[i,:] > thresh

            # # perform skeletonization
            # image = skeletonize(image)

            #self.mnist.test.images[i,:] = numpy.reshape(image, 784)

    def augmentdata(self):
        # Do augmentations
        count = 0
        for i in range(len(self.mnist.train.images)):
            #for r in range(4):
            self.augmentedimages[i,:] = copy.deepcopy(self.mnist.train.images[i,:])
            self.augmentedlabels[i,:] = copy.deepcopy(self.mnist.train.labels[i,:])
            #
            # rotimg=rotate(numpy.reshape(self.mnist.train.images[i,:], (28, 28)), r-2, reshape=False)
            # self.augmentedimages[count,:] = numpy.reshape(rotimg, 784)
            #
            # count = count + 1

            #image = copy.deepcopy(numpy.reshape(self.mnist.train.images[i,:], (28, 28)))

            # thresh = threshold_mean(self.augmentedimages[i,:])
            # self.augmentedimages[i,:] = self.augmentedimages[i,:] > thresh

            # # # perform skeletonization
            # # image = skeletonize(image)
            #
            #self.augmentedimages[i,:] = numpy.reshape(image, 784)
            # self.augmentedlabels[i,:] = copy.deepcopy(self.mnist.train.labels[i,:])

    def __next_batch(self, batch_size):
        raise NotImplemented

    def train_next_batch(self, batch_size):
        # samples = random.sample(range(len(self.mnist.train.images)),  batch_size)
        # return (self.mnist.train.images[samples,:], self.mnist.train.labels[samples,:])

        samples = random.sample(range(len(self.augmentedimages)),  batch_size)
        return (self.augmentedimages[samples,:], self.augmentedlabels[samples,:])

        # choice = random.randint(0,1)
        # if choice==1:
        #     samples = random.sample(range(len(self.mnist.train.images)),  batch_size)
        #     return (self.mnist.train.images[samples,:], self.mnist.train.labels[samples,:])
        # else:
        #     samples = random.sample(range(len(self.augmentedimages)),  batch_size)
        #     return (self.augmentedimages[samples,:], self.augmentedlabels[samples,:])

        #return self.__next_batch(batch_size)

if __name__ == "__main__":

    dataset = mnistdata()
    dataset.augmentdata()
    dataset.datachange()

    batch = dataset.train_next_batch(4)

    #batch = dataset.mnist.train.next_batch(4)

    print(batch[0])
    print(type(batch[0]))

    #
    # batch = dataset.mnist.train.next_batch(2)

    # print(type(batch))
