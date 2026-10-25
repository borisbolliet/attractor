import os, subprocess, sys
LIMIT = float(os.environ.get("ATTRACTOR_T1", "40"))
os.environ["ATTRACTOR_T1"] = "80"
cmd = [sys.executable, "-c", "import os; print(os.environ['ATTRACTOR_T1'])"]
child = subprocess.run(cmd, capture_output=True, text=True)
print(LIMIT, child.stdout.strip())
