from engine import Value
#testing pow with large numbers
a = Value([1e4, 1e5, 1e6])
b = Value([2.0, 2.5, 3.0])
c = a ** b
c.backward()
print(a)
print(b)
print(c)