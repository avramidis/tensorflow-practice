# @authors: Eleftherios Avramidis
# @email: el.avramidis@gmail.com
# @date: 2017/08/26
# @copyright: MIT License

import numpy
import random
import copy

from data import Data

#from tensorflow.examples.tutorials.mnist import input_data

# from skimage.filters import threshold_mean
# from skimage.morphology import skeletonize

# from scipy.ndimage import rotate


class MnistData(Data):
    """
    docstring here
        :param Data: Class Data.
    """

    mnist = []            # Variable with the MNIST data

    trainimages = []

    validationimages = []

    testimages = []

    augmentedimages = []
    augmentedlabels = []

    # Default constructor.
    #
    #  Sets the initial values of the object.
    #
    #  @param self                      is the reference to the current object.
    def __init__(self):
        """
        Initialisation of the training and validation data.
        Sets the initial values of the object.
            :param self: Reference to the current object.
        """

        train_data = numpy.genfromtxt(
            'train.csv', delimiter=',', skip_header=True)
        train_labels = numpy.zeros((len(train_data), 10))

        print("Size of training data: ", len(train_data))

        for i in range(0, len(train_data)):
            train_labels[i, int(train_data[i, 0])] = 1
        train_data = train_data[:, 1:]

        # Split train data to validation data
        samples = random.sample(range(len(train_data)), 4200)
        validation_data = train_data[samples, :]
        validation_labels = train_labels[samples, :]

        train_data = numpy.delete(train_data, samples, 0)
        train_labels = numpy.delete(train_labels, samples, 0)

        # Read test data
        test_data = numpy.genfromtxt(
            'test.csv', delimiter=',', skip_header=True)
        test_labels = numpy.zeros((len(test_data), 10))

        print("Size of test data: ", len(test_data))

        for i in range(0, len(test_data)):
            test_labels[i, int(test_data[i, 0])] = 1
        test_data = test_data[:, 1:]

        super().__init__(train_data=train_data,
                         train_labels=train_labels,
                         validation_data=validation_data,
                         validation_labels=validation_labels,
                         test_data=test_data,
                         test_labels=test_labels)

    # def datachange(self):
    #     for i in range(len(self.mnist.validation.images)):
    #         #image = copy.deepcopy(numpy.reshape(self.mnist.validation.images[i,:], (28, 28)))
    #         thresh = threshold_mean(self.mnist.validation.images[i, :])
    #         self.mnist.validation.images[i,
    #                                      :] = self.mnist.validation.images[i, :] > thresh

    #         # # perform skeletonization
    #         # image = skeletonize(image)

    #         #self.mnist.validation.images[i,:] = numpy.reshape(image, 784)

    #     for i in range(len(self.mnist.test.images)):
    #         #image = copy.deepcopy(numpy.reshape(self.mnist.test.images[i,:], (28, 28)))
    #         thresh = threshold_mean(self.mnist.test.images[i, :])
    #         self.mnist.test.images[i,
    #                                :] = self.mnist.test.images[i, :] > thresh

    #         # # perform skeletonization
    #         # image = skeletonize(image)

    #         #self.mnist.test.images[i,:] = numpy.reshape(image, 784)

    # def augmentdata(self):
    #     # Do augmentations
    #     count = 0
    #     for i in range(len(self.mnist.train.images)):
    #         # for r in range(4):
    #         self.augmentedimages[i, :] = copy.deepcopy(
    #             self.mnist.train.images[i, :])
    #         self.augmentedlabels[i, :] = copy.deepcopy(
    #             self.mnist.train.labels[i, :])
    #         #
    #         # rotimg=rotate(numpy.reshape(self.mnist.train.images[i,:], (28, 28)), r-2, reshape=False)
    #         # self.augmentedimages[count,:] = numpy.reshape(rotimg, 784)
    #         #
    #         # count = count + 1

    #         #image = copy.deepcopy(numpy.reshape(self.mnist.train.images[i,:], (28, 28)))

    #         # thresh = threshold_mean(self.augmentedimages[i,:])
    #         # self.augmentedimages[i,:] = self.augmentedimages[i,:] > thresh

    #         # # # perform skeletonization
    #         # # image = skeletonize(image)
    #         #
    #         #self.augmentedimages[i,:] = numpy.reshape(image, 784)
    #         # self.augmentedlabels[i,:] = copy.deepcopy(self.mnist.train.labels[i,:])


if __name__ == "__main__":

    dataset = MnistData()

    batch_xs, batch_ys = dataset.train_random_batch(2)

    print(batch_xs)

    # dataset.augmentdata()
    # dataset.datachange()

    # batch = dataset.train_next_batch(4)

    # batch = dataset.mnist.train.next_batch(4)

    # print(batch[0])
    # print(type(batch[0]))

    #
    # batch = dataset.mnist.train.next_batch(2)

    # print(type(batch))
