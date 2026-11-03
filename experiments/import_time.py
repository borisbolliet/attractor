"""A decorator's body runs at import, the wrapper's body at every call.

Run from the project root:
    python -c "import experiments.import_time as m; m.area(2, 3); m.area(4, 5); import experiments.import_time"
"""
import functools

LOUD_REGISTRY = {}


def loud_register(name):
    def deco(f):
        print(f"decorator: registering {name}")  # runs once, when the module is imported
        LOUD_REGISTRY[name] = f

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f"wrapper: calling {name}")  # runs at every call
            return f(*args, **kwargs)
        return wrapper
    return deco


@loud_register("area")
def area(w, h):
    return w * h
