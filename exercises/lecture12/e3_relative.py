"""Run from the project root: python exercises/lecture12/e3_relative.py
Writes a small package into a temporary folder and runs five commands there."""
import subprocess, sys, tempfile
from pathlib import Path

FILES = {
    "geometry/__init__.py": "",
    "geometry/units.py": "MM_PER_CM = 10\n",
    "geometry/shapes.py": "from .units import MM_PER_CM\n"
                          "print('30 cm =', 30 * MM_PER_CM, 'mm')\n",
    "scripts/area.py": "import geometry.shapes\n",
}
COMMANDS = ["python geometry/shapes.py", "python -m geometry.shapes", "python scripts/area.py",
            "python -m scripts.area", "cd scripts && python -m area"]
work = Path(tempfile.mkdtemp())
for name, text in FILES.items():
    (work / name).parent.mkdir(exist_ok=True)
    (work / name).write_text(text)
for command in COMMANDS:
    run = subprocess.run(command.replace("python", sys.executable), shell=True, cwd=work,
                         capture_output=True, text=True)
    print(f"{command:<30} {(run.stdout + run.stderr).strip().splitlines()[-1]}")
