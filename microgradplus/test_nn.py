import numpy as np
from engine import Value
from nn import Linear, ReLU, Tanh, Sequential

# define a model
model = Sequential(
    Linear(2, 4),
    Tanh(),
    Linear(4, 2)
)

# random input data
x = Value(np.random.randn(4, 2))  # bs = 4
print(f"x = {x.data}")

# fp
out = model(x)
print(f"out = {out.data}")
