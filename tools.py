"""Decorators of the project: call counting, timing and a registry of model factories."""
import functools
import time


def counted(f):
    """Return f wrapped so that `wrapper.calls` counts its calls."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    def reset():
        wrapper.calls = 0

    wrapper.calls = 0
    wrapper.reset = reset
    return wrapper


def timed(f):
    """Return f wrapped so that `wrapper.seconds` accumulates the wall time of its calls and `wrapper.calls` their number."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        wrapper.seconds += time.perf_counter() - start  # the wrapper's own overhead is included
        wrapper.calls += 1
        return result

    wrapper.seconds = 0.0
    wrapper.calls = 0
    return wrapper


REGISTRY = {}


def register(name):
    """Decorator factory: store the decorated factory under `name` in REGISTRY and return it unchanged."""
    def deco(factory):
        REGISTRY[name] = factory
        return factory
    return deco


def make_model(name, **params):
    """Build a right-hand side from a registered factory name."""
    return REGISTRY[name](**params)
