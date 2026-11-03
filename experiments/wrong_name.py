"""Puzzle: a wrapper that names a module-level function instead of its parameter."""


def double(x):
    return 2 * x


def square(x):
    return x * x


f = double


def counted(g):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


d, s = counted(double), counted(square)
print(d(3), s(3))
print(s.__code__.co_freevars)
