# @authors: Eleftherios Avramidis
# @email: el.avramidis@gmail.com
# @date: 2017/10/28
# @copyright: MIT License

import random
import numpy
import numpy.random

# from skimage.filters import threshold_mean
# from skimage.morphology import skeletonize

# from scipy.ndimage import rotate


class Data(object):
    """
    The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

        :param object: object class.
    """
    train_data = []
    """Docstring for class variable Data.train_data"""
    train_labels = []
    """Docstring for class variable Data.train_labels"""

    validation_data = []
    """Docstring for class variable Data.test_data"""
    validation_labels = []
    """Docstring for class variable Data.test_labels"""

    test_data = []
    """Docstring for class variable Data.test_data"""
    test_labels = []
    """Docstring for class variable Data.test_labels"""

    def __init__(self,
                 train_data,
                 train_labels,
                 validation_data,
                 validation_labels,
                 test_data, test_labels):
        """
        Default constructor.
        Sets the initial values of the object.
            :param self: Reference to the current object.
            :param train_data: Training data.
            :param train_labels: Training labels.
            :param validation_data: Validation data.
            :param validation_labels: Validation labels.
            :param test_data: Test data.
            :param test_labels: Test labels.
        """

        self.train_data = train_data
        self.train_labels = train_labels

        self.validation_data = validation_data
        self.validation_labels = validation_labels

        self.test_data = test_data
        self.test_labels = test_labels

    def train_random_batch(self, batch_size):
        """
        docstring here
            :param self: Reference to the current object.
            :param batch_size: Size of the batch to be selected and returned.
        """

        samples = random.sample(range(len(self.train_data)), batch_size)
        return (self.train_data[samples, :], self.train_labels[samples, :])


# if __name__ == "__main__":

#     train_data = numpy.random.rand(100, 784)
#     train_labels = numpy.random.rand(100, 10)
#     test_data = numpy.random.rand(10, 784)
#     test_labels = numpy.random.rand(10, 10)

#     data = Data(train_data=train_data,
#                 train_labels=train_labels,
#                 test_data=test_data,
#                 test_labels=test_labels)

#     batch = data.train_random_batch(2)

#     print(batch)
