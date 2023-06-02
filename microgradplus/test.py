from engine import Value

a = Value([1, 2, 3])
c = Value.abs(a)
c.backward()
print(f"a = {a}")
print(f"c = abs(a) = {c}")