from engine import Value

a = Value(-2)
b = Value(3)
c = a + b
d = c * b
e = d / a
f = e - b
g = f ** 2
h = Value.sqrt(g)
i = Value.tanh(h)

i.backward()

print(f"a = {a}")
print(f"b = {b}")
print(f"c = a + b = {c}")
print(f"d = c * b = {d}")
print(f"e = d / a = {e}")
print(f"f = e - b = {f}")
print(f"g = f ** 2 = {g}")
print(f"h = sqrt(g) = {h}")
print(f"i = tanh(h) = {i}")