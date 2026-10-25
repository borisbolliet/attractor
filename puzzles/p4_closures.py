from functools import partial
rates = [lambda x: r * x for r in (0.5, 0.8, 1.1)]
print([f(1.0) for f in rates])
rates = [lambda x, r=r: r * x for r in (0.5, 0.8, 1.1)]
print([f(1.0) for f in rates])
def scale(r, x):
    return r * x
rates = [partial(scale, r) for r in (0.5, 0.8, 1.1)]
print([f(1.0) for f in rates])
def make_rate(r):
    return lambda x: r * x
rates = [make_rate(r) for r in (0.5, 0.8, 1.1)]
print([f(1.0) for f in rates])
