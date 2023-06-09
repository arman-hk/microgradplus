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

class ReLU:
    def __call__(self, x):
        return x.relu()

class Sequential:
    def __init__(self, *layers):
        # stores layers
        self.layers = layers

    def __call__(self, x):
        # forward pass on each layer with the output of the prev layer
        for layer in self.layers:
            x = layer(x)
        return x
    def backward(self, grad):
        # backward pass on each layer but in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
