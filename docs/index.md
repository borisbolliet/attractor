# attractor

Predator–prey models, fixed-step integrators and cycle periods: the running
example of the C1 Research Computing course.

## Installation

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Example

```python
from attractor.integrators import RK4
from attractor.models import LotkaVolterra

traj = RK4(0.05).run(LotkaVolterra(), [1.0, 0.5], 0.0, 40.0)
```

The [API reference](api.md) is generated from the docstrings of the package.
