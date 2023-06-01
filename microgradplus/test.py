from engine import Value
a = Value([1, 2, 3])
c = Value.sqrt(a)
c.backward()
print(f"a = {a}")
print(f"c = a/b = {c}")