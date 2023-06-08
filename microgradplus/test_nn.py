import numpy as np
from engine import Value
from nn import Linear

input_dim, output_dim = 5, 3
layer = Linear(input_dim, output_dim)

# random input data
x = Value(np.random.randn(10, input_dim))  # bc = 10

# fp
out = layer(x)
print(f"out = {out.data}")

# bp
grad = Value(np.random.randn(10, output_dim))
dx = layer.backward(grad)

print(f"dx = {dx}")

# grads of the parameters
print(f"layer.weights.grad: {layer.weights.grad}")
print(f"layer.bias.grad: {layer.bias.grad}")