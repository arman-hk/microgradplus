import numpy as np

class Value:
    """A node in the computation graph."""
    def __init__(self, data, _children=(), _grad_fn=None):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        # internal variables
        self._backward = lambda: None
        self._prev = set(_children)  # stores previous objects
        self._grad_fn = _grad_fn  # stores the gradient function of the operation