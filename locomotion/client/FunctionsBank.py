# All necessary libraries.
import math


# This class works as a bank of activation functions for the evolutionary process of neural networks.
class FunctionBank:

    # Constructor
    def __init__(self):

        pass

    @staticmethod
    def negative_sin(z):

        return -math.sin(z)

    @staticmethod
    def negative_abs(z):

        return -abs(z)

    @staticmethod
    def negative_square(z):

        return -(z ** 2)

    @staticmethod
    def square_abs(z):

        return math.sqrt(abs(z))

    @staticmethod
    def negative_square_abs(z):
        return -math.sqrt(abs(z))
