from engine import Value

a = Value([1, 2, 3])
b = Value([4, 5, 6])
c = Value([7, 8, 9])
d = b - a
e = d - c
e.backward()

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = b - a = {d}")
print(f"e = d - c = {e}")