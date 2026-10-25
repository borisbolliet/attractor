import importlib, os, sys
print("stepsize" in sys.modules)
from stepsize import H
import stepsize
print(stepsize is sys.modules["stepsize"], H)
stepsize.H = 0.02
print(H, stepsize.H)
os.environ["ATTRACTOR_H"] = "0.01"
m = importlib.reload(stepsize)
print(m is stepsize, stepsize.H, stepsize.STEPS, H)
