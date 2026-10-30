"""Three models with different growth rates: bare lambdas against the factory (run from the project root as python -m puzzles.model_loop)."""
import numpy as np

from attractor.cycles import cycle_period
from attractor.integrators import integrate, rk4_step
from attractor.models import make_lotka_volterra

y = np.array([1.0, 0.0])  # at z = 0 the first component of rhs is a*x = a
models = [lambda t, y: np.array([a*y[0] - 0.08*y[0]*y[1],
                                 0.77*y[0]*y[1] - 0.77*y[1]])
          for a in (0.5, 0.8, 1.1)]
print(" ".join(f"{f(0.0, y)[0]:g}" for f in models))
models = [make_lotka_volterra(a, 0.08, 0.77, 0.77) for a in (0.5, 0.8, 1.1)]
print(" ".join(f"{f(0.0, y)[0]:g}" for f in models))
print([f.__closure__[0].cell_contents for f in models])
for f in models:
    t, traj = integrate(rk4_step, f, np.array([1.5, 8.0]), 0.0, 80.0, 0.05)
    print(f"a = {f.__closure__[0].cell_contents:g}: period {cycle_period(t, traj[:, 0])[0]:.2f} days")
