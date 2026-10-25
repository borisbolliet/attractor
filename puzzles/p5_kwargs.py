def rhs(t, y, *args, **kwargs):
    return t, y, args, kwargs
print(rhs(0.0, 1.0))
print(rhs(0.0, 1.0, 0.8, 0.4, h=0.05))
p = {"a": 0.8, "b": 0.08}
print(rhs(0.0, 1.0, **p))
def f(a, b=2, *, c):
    return a, b, c
print(f(1, c=3), f.__defaults__, f.__kwdefaults__)
