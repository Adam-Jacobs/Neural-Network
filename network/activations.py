import numpy as np


class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(x, 0)


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
