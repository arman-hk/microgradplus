#### [engine.py](/microgradplus/engine.py)
- [x] `_grad_fn`: to store the gradient calculation logic specific to the operation that produced the `Value`
- [x] `__repr__`: repr operator to display the data and it's gradient when printed
- [x] `__add__`: addition operator with it's grad func(works on scalars, and arrays)
- [x] `Context`: ctx to store the arrays for backprop
- [x] `backward`: backward pass of the computation, calculating gradients for all nodes in the computation graph by chain rule
- [x] `__mul__`: element-wise multiplication operator with it's grad func(works on scalars, and arrays)
- [x] `__pow__`: fast power function with it's grad func(works on scalars, and arrays)
- [x] `__neg__`: negation operator with it's grad func(works on scalars, and arrays)
- [x] `__sub__`: subtraction operator with it's grad func(works on scalars, and arrays)
- [x] `__truediv__`: true division operator with it's grad func(works on scalars, and arrays)
- [x] `sqrt`: square root fucntion with it's grad func(works on scalars, and arrays)
- [x] `exp`: exponential function with it's grad func(works on scalars, and arrays)
- [x] `log`: natural logarithm function with it's grad func(works on scalars, and arrays)
- [x] `abs`: absolute function with it's grad func(works on scalars, and arrays)
- [x] `relu`: rectified linear unit function with it's grad func(works on scalars, and arrays)
- [x] `sigmoid`: element-wise sigmoid function with it's grad func(works on scalars, and arrays)
- [x] `tanh`: hyperbolic tangent function with it's grad func(works on scalars, and arrays)
- [x] `__matmul__`: matrix multiplication @ function with it's grad func(works on arrays)
- [x] `__rmatmul__`: reverse matrix multiplication @ function
- [x] `T`: matrix transpose fucnction

#### [nn.py](/microgradplus/nn.py)
- [x] `Linear`: a linear layer in a neural network. It has a forward pass that computes the dot product of the inputs with the weights and adds the bias, and a backward pass that computes the gradients with respect to the inputs and parameters.
- [x] `Sequential`: an ordered sequence of neural network layers, that simplifies the process of defining, and training a neural network by providing forward and backward passes through the entire network.