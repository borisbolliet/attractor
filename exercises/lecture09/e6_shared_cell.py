import numpy as np
def make_growth(params):
    def rhs(t, y):
        return params[0] * y
    return rhs
params = [0.8, 0.08]
f, g = make_growth(params), make_growth(params)
params[0] = 1.6
print(f"{f(0.0, np.array([1.0]))[0]:g} {g(0.0, np.array([1.0]))[0]:g}")
print(f.__closure__[0].cell_contents is g.__closure__[0].cell_contents)
params = [0.8, 0.08]
f, g = make_growth(tuple(params)), make_growth(list(params))
params[0] = 1.6
print(f"{f(0.0, np.array([1.0]))[0]:g} {g(0.0, np.array([1.0]))[0]:g}")
