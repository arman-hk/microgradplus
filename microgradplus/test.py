from engine import Value

a = Value([4.0, 10.0, 12.0])
b = Value([2.0, 5.0, 6.0])
c = a / b
c.backward()

print(f"a = {a}")
print(f"b = {b}")
print(f"c = a/b = {c}")