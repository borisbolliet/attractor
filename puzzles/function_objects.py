"""The cells of the closure that make_lotka_volterra returns (run from the project root as python -m puzzles.function_objects)."""
from models import make_lotka_volterra

rhs = make_lotka_volterra(0.8, 0.08, 0.77, 0.77)
print(rhs.__qualname__)
print(rhs.__defaults__, rhs.__code__.co_freevars)
print([c.cell_contents for c in rhs.__closure__])
