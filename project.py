import numpy as np


class errors:

    @staticmethod
    def mse(real, predicted):
        return np.sqrt(np.square(np.subtract(real, predicted)).mean())


class activations:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
