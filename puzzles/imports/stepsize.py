import os
print("stepsize.py runs")
T = 40.0
H = float(os.environ.get("ATTRACTOR_H", "0.05"))
STEPS = round(T / H)
