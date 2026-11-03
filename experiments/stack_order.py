"""Order of stacked decorators, and the attribute that `wraps` copies once.

Run from the project root: python -m experiments.stack_order
"""
from tools import counted, timed


def A(f):
    print("A applied to", f.__name__)
    return f


def B(f):
    print("B applied to", f.__name__)
    return f


@A
@B
def g(x):
    return x


h = counted(timed(g))
for _ in range(3):
    h(1)
print(h.calls, h.seconds, h.__wrapped__.calls, h.__wrapped__.seconds > 0)
print(h.__wrapped__.__wrapped__ is g)
