"""Run from the project root: python exercises/lecture12/e5_stale_metadata.py
Installs a copy of the project, editable, into a new venv; changes the version, adds a
function and its console command; runs three checks before and after reinstalling."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp())
project = work / "attractor"
shutil.copytree(".", project, ignore=shutil.ignore_patterns(".venv*", "results*", "build", "dist",
                                                            "*.egg-info", ".git"))
subprocess.run([sys.executable, "-m", "venv", work / "env"], check=True)
py, bin_ = work / "env/bin/python", work / "env/bin"
pip = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
subprocess.run(pip + ["numpy"], check=True)
subprocess.run(pip + ["-e", project, "--no-deps"], check=True)

toml = project / "pyproject.toml"
text = toml.read_text().replace('version = "1.0.1"', 'version = "1.1.0"')
text = text.replace('attractor-fit = "attractor.cli:fit"',
                    'attractor-fit = "attractor.cli:fit"\nattractor-version = "attractor.cli:show_version"')
toml.write_text(text)
with open(project / "src/attractor/cli.py", "a") as fh:
    fh.write('\n\ndef show_version():\n    from . import __version__\n    print("attractor", __version__)\n')

CHECKS = {"import attractor": [py, "-c", "import attractor; print(attractor.__version__)"],
          "cli.show_version()": [py, "-c", "from attractor.cli import show_version; show_version()"],
          "attractor-version": [bin_ / "attractor-version"]}


def run(label):
    for name, command in CHECKS.items():
        try:
            out = subprocess.run(command, capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            out = "command not found"
        print(f"{label:<16} {name:<20} {out}")


run("before")
subprocess.run(pip + ["-e", project, "--no-deps"], check=True)
run("after reinstall")
