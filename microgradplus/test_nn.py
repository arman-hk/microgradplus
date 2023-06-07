import numpy as np
from engine import Value
from nn import Linear

input_dim, output_dim = 5, 3
layer = Linear(input_dim, output_dim)

# random input data
x = Value(np.random.randn(10, input_dim))  # bc = 10

# fp
out = layer(x)
print(f"out: {out.data}")