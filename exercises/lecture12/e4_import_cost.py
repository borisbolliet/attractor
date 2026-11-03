"""Run from the project root, in an environment where attractor is installed:
python exercises/lecture12/e4_import_cost.py
Cumulative import times, in milliseconds, from python -X importtime."""
import subprocess, sys

for statement in ["import numpy", "import attractor", "import attractor.cli",
                  "import scipy.optimize"]:
    run = subprocess.run([sys.executable, "-X", "importtime", "-c", statement],
                         capture_output=True, text=True, check=True)
    rows = [line.split("|") for line in run.stderr.splitlines()[1:]]   # self | cumulative | name
    times = {name.strip(): int(cumulative) / 1000 for _, cumulative, name in rows}
    print(f"{statement:<22} {max(times.values()):6.1f} ms   scipy imported: {'scipy' in times}")
