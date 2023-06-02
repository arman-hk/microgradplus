from engine import Value
import numpy as np
a = Value([1, 2, 3])
c = Value.log(a)
c.backward()
print(f"a = {a}")
print(f"c = log(a) = {c}")