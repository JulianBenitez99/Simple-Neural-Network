
import numpy as np

inputs = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
])


class activations:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


activationss = activations
print(activationss.sigmoid(inputs[0][2]))
print(activationss.sigmoid(inputs))
print(activationss.sigmoid_derivative(inputs))
