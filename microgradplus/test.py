from engine import Value
a = Value([1, 2, 3])
c = Value.exp(a)
c.backward()
print(f"a = {a}")
print(f"c = exp(a) = {c}")