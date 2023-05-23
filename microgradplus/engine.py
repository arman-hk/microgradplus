import numpy as np

class Value:
    """A node in the computation graph."""
    
    def __init__(self, data, _children=(), _grad_fn=None):
        self.data = np.array(data)
        self.grad = None
        # internal variables
        self._prev = set(_children)  # stores previous objects
        self._grad_fn = _grad_fn  # stores the gradient function
    
    def __add__(self, other):
        # if `other` is a scalar, convert it to a `Value`
        if isinstance(other, (int, float)):
            other = Value(other)
        
        def _grad_fn():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

            
        out = Value(self.data + other.data, (self, other), _grad_fn)
        return out
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
