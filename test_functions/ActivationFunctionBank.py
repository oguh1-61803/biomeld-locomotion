# All necessary libraries.
from scipy.special import erf
import math

# This class works as a bank of activation functions for the evolutionary process of neural networks.
class ActivationFunctionBank:

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

    @staticmethod
    def cos(z):

        return math.cos(z)

    @staticmethod
    def negative_cos(z):

        return -math.cos(z)

    @staticmethod
    def gelu(z):
        cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))

        return z * cdf
