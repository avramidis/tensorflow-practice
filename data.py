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
    test_data = []
    """Docstring for class variable Data.test_data"""
    test_labels = []
    """Docstring for class variable Data.test_labels"""

    def __init__(self):
        """
        Default constructor.
        Sets the initial values of the object.
            :param self: Reference to the current object.
        """

        self.train_data = numpy.random.rand(100, 784)
        self.train_labels = numpy.random.rand(100, 10)

        self.test_data = numpy.random.rand(10, 784)
        self.test_labels = numpy.random.rand(10, 10)

    def train_random_batch(self, batch_size):
        """
        docstring here
            :param self: Reference to the current object.
            :param batch_size: Size of the batch to be selected and returned.
        """

        samples = random.sample(range(len(self.train_data)), batch_size)
        return (self.train_data[samples, :], self.train_data[samples, :])


if __name__ == "__main__":
    
    data = Data()

    batch = data.train_random_batch(2)
    print(batch)
