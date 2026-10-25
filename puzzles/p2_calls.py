def step(y, h, log=[]):
    log.append(h)
    return y + h, log
print(step(0.0, 0.1)[1])
print(step(0.0, 0.2)[1])
def bump(v, w):
    v += w
def regrow(v):
    v = v + [1]
u = [0]; bump(u, [1]); regrow(u); print(u)
t = (0,); bump(t, (1,)); print(t)
n = 0; bump(n, 1); print(n)
