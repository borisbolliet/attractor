h = 0.05
def show():
    print(h)
def shrink():
    h = h / 2
    return h
show()
try:
    shrink()
except UnboundLocalError as e:
    print("UnboundLocalError:", e)
def counter():
    n = 0
    def tick():
        nonlocal n
        n += 1
        return n
    return tick
tick = counter()
print(tick(), tick(), counter()())
