import numpy as np

class Context:
    """Array info for backprop"""
    
    def __init__(self):
        self.saved_arrays = {}

    def save_for_backward(self, *keys):
        for idx, key in enumerate(keys):
            self.saved_arrays[idx] = key

class Value:
    """A node in the computation graph"""
    
    def __init__(self, data, _children=(), _grad_fn=None):
        self.data = np.array(data)
        self.grad = None
        # internal variables
        self._prev = set(_children)
        self._grad_fn = _grad_fn
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        ctx = Context()
        ctx.save_for_backward(self.data, other.data)
        
        def _grad_fn():
            self_data, other_data = ctx.saved_arrays.values()
            if self.grad is None:
                self.grad = np.zeros_like(self_data)
            if other.grad is None:
                other.grad = np.zeros_like(other_data)
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
    
        out = Value(self.data + other.data, (self, other), _grad_fn)
        return out
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
