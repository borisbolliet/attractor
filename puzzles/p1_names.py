a = [1, 2, 3]
b = a
b.append(4)
print(a, a is b)
c = a[:]
print(c == a, c is a)
x = 1000; y = 1000
print(x == y, x is y)
