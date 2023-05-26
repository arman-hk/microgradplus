from engine import Value

a = Value([1, 2, 3])
b = Value([4, 5, 6])
c = a + b
c.backward()
print(a)
print(b)
print(c)