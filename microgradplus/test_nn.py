import numpy as np
from engine import Value
from nn import Linear, ReLU, Sequential

# define a model
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 2)
)

# random input data
x = Value(np.random.randn(4, 2))  # bc = 4
print(f"x = {x.data}")

# fp
out = model(x)
print(f"out = {out.data}")
