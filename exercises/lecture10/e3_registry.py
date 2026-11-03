"""Run from the project root: python -m exercises.lecture10.e3_registry"""
import importlib
from attractor.tools import REGISTRY
print(REGISTRY)
from attractor import models
print(sorted(REGISTRY),
      id(REGISTRY["lotka_volterra"]) == id(models.make_lotka_volterra))
m = importlib.reload(models)
print(sorted(REGISTRY), REGISTRY["lotka_volterra"] is models.make_lotka_volterra)
