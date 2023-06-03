from engine import Value

a = Value(-2)
b = Value(3)
c = a + b
d = c * b
e = d / a
f = e - b
g = f ** 2
h = g.exp()
i = Value.sqrt(h)
j = i.abs()
k = Value.relu(j)

k.backward()

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
print(f"e = {e}")
print(f"f = {f}")
print(f"g = {g}")
print(f"h = {h}")
print(f"i = {i}")
print(f"j = {j}")
print(f"k = {k}")