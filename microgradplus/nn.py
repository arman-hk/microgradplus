import numpy as np
from engine import Value

class Linear:
    def __init__(self, input_dim, output_dim):
        # init weights and biases
        self.weights = Value(np.random.randn(input_dim, output_dim) * 0.01)
        self.bias = Value(np.zeros(output_dim))

    def __call__(self, x):
        # forward pass
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad):
        # grads with respect to inputs and params
        self.weights.grad += self.x.T.data @ grad.data
        self.bias.grad += np.sum(grad.data, axis=0)
        return grad @ self.weights.T
