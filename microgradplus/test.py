from engine import Value

a = Value([-1, 0, 1, 2, 3])
c = Value.relu(a)
c.backward()

print(f"a = {a}")
print(f"c = relu(a) = {c}")