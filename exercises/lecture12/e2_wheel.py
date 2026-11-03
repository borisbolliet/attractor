"""Run from the project root: python exercises/lecture12/e2_wheel.py
(a) builds the wheel of a copy of the project with four files added (pip fetches setuptools);
(b) checks four wheel names against the tags that two interpreters accept."""
import shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

ADDED = ["src/attractor/data/C0.csv", "src/attractor/data/notes.txt", "tests/test_cycles.py",
         "docs/index.md"]
work = Path(tempfile.mkdtemp()) / "attractor"
shutil.copytree(".", work, ignore=shutil.ignore_patterns(".venv*", "results*", "build", "dist",
                                                         "*.egg-info", ".git"))
for name in ADDED:
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    (work / name).write_text("\n")
subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", "dist", "-q"],
               cwd=work, check=True)
(wheel,) = (work / "dist").glob("*.whl")
names = zipfile.ZipFile(wheel).namelist()
print(wheel.name)
for name in ADDED + ["src/attractor/datasets.py", "data/C9.csv", "simulate.py", "README.md"]:
    print(f"{name:<30} {'in' if name.removeprefix('src/') in names else 'not in'}")

WHEELS = ["attractor-1.0.1-py3-none-any.whl", "tornado-6.5.10-cp39-abi3-macosx_10_9_universal2.whl",
          "psutil-7.2.2-cp36-abi3-macosx_11_0_arm64.whl",
          "debugpy-1.8.22-cp312-cp312-macosx_15_0_universal2.whl"]
for python, macos, arch in [((3, 13), (15, 0), "arm64"), ((3, 12), (13, 0), "x86_64")]:
    plats = list(mac_platforms(macos, arch))
    ok = set(cpython_tags(python, platforms=plats)) | set(compatible_tags(python, platforms=plats))
    print(f"CPython {python[0]}.{python[1]}, macOS {macos[0]}, {arch}:",
          [w.split("-")[0] for w in WHEELS if parse_wheel_filename(w)[3] & ok])
