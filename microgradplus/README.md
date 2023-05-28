#### [engine.py](/microgradplus/engine.py)
- [x] `_grad_fn`: to store the gradient calculation logic specific to the operation that produced the `Value`
- [x] `__repr__`: repr function to display the data and it's gradient when printed
- [x] `__add__`: add function with it's grad function
- [x] `Context`: ctx to store the arrays for backprop
- [x] `backward`: backward pass of the computation, calculating gradients for all nodes in the computation graph by chain rule
- [x] `__mul__`: element-wise multiplication works for both scalars and matrices (relu, dropout)
- [x] `__pow__`: fast power function that works between all type of inputs(check [test](/microgradplus/test.py))