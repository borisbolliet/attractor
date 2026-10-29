import copy
import functools
import numpy as np
def cached_copy(f):
    cached = functools.lru_cache(maxsize=None)(f)
    @functools.wraps(f)
    def wrapper(*args):
        return copy.deepcopy(cached(*args))
    wrapper.cache_info = cached.cache_info
    return wrapper
@cached_copy
def grid(n):
    return np.linspace(0.0, 1.0, n)
a, b = grid(5), grid(5)
print(a is b, np.shares_memory(a, b), grid.cache_info())
