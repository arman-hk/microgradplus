import numpy as np
from engine import Value
from nn import Linear, Tanh, Sequential, MAE

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

# random targets
targets = Value(np.random.randn(4, 2)) 

# compute loss
mse = MAE()
loss = mse(out, targets)
print(f"loss = {loss.data}")