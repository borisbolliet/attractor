class Grid:
    step = 0.1
a, b = Grid(), Grid()
a.step = 0.05
print(a.step, b.step, "step" in vars(a), "step" in vars(b))
del a.step
print(a.step, b.step, "step" in vars(a))
