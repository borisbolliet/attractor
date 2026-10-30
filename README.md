# attractor

Running example of the C1 Research Computing course (MPhil in Data Intensive
Science, University of Cambridge), from Lecture 4, "Reproducible Python
environments", onwards. The project measures the cycle period of ten chemostat
predator-prey experiments and is small enough to be rebuilt from scratch in a
few minutes on a laptop or on a Linux workstation.

## Lecture states

The state of the project at each lecture is a tag, checked out with
`git switch --detach <tag>`; `git switch main` returns to the latest state.

| Tag | State |
|---|---|
| `lecture-09-start` | start of Lecture 9: model, integrators, driver, puzzles and exercises; `rk4_step` writes the new state into its argument |
| `lecture-09` | end of Lecture 9: `rk4_step` rebinds `y` instead (`git diff lecture-09-start lecture-09`) |
| `lecture-10` | end of Lecture 10: decorators in `tools.py`, a registry of models, experiments and exercises |
| `lecture-11` | end of Lecture 11: model classes, the class `Trajectory`, integrator classes and the lynx–hare fit |
| `lecture-12-start` | start of Lecture 12: the modules as the package `src/attractor`, `pyproject.toml` (version 1.0.0) and three console commands; `lynx_hare.csv` is not declared as package data, so the wheel leaves it out |
| `lecture-12` | end of Lecture 12: `lynx_hare.csv` declared as package data, version 1.0.1 (`git diff lecture-12-start lecture-12`) |

## Setup

```bash
git clone https://github.com/borisbolliet/attractor.git ~/GitHub/attractor
cd ~/GitHub/attractor
source setup_env.sh          # Python 3.13 venv, pinned packages, the project, .env, kernel attractor-py313
attractor-periods            # table of periods and results/periods.csv
python ask_llm.py            # optional: two sentences about the table from a language model
```

The clone goes to `~/GitHub/attractor` so that it does not collide with the
`~/attractor` directory of the first lectures of the course. `source
setup_env.sh` typically takes 10 to 15 s with a warm pip cache (lecture
laptop, Python 3.13.11). A second `source
setup_env.sh` only prints the environment report, unless `pyproject.toml` has
changed, in which case it reinstalls the project. The script must be sourced:
run as `./setup_env.sh` it stops with a one-line message, because a script
that runs in a child shell cannot activate a venv in the calling shell.

Python 3.13 is expected as `python3.13`; another interpreter is selected with
`PYTHON=/path/to/python3.13 source setup_env.sh`. Obtaining 3.13: macOS,
`brew install python@3.13`; any system, `uv python install 3.13` then
`PYTHON="$(uv python find 3.13)" source setup_env.sh`. A `.venv` built with
another Python version, such as the 3.12 one of Lectures 4 to 9, is rebuilt.

`ask_llm.py` needs an OpenAI-compatible chat endpoint in `LLM_BASE_URL`: the
course endpoint given in the lecture, a local `llama-server` or Ollama
(`http://localhost:11434/v1`), or `https://api.openai.com/v1` with a real
`LLM_API_KEY` and `LLM_MODEL`. The value is written in `.env`. After editing
`.env`, run `source setup_env.sh` again: the shell still holds the exported
placeholder and `load_dotenv()` does not override it. With the placeholder
left in place the script stops with `ask_llm.py: cannot reach
http://localhost:8080/v1; set LLM_BASE_URL in .env and run source setup_env.sh again`.

## Environment files

1. **Interpreter and packages.** `requirements.txt` gives package
   constraints (`numpy>=1.26`, `matplotlib>=3.8`, ...). `requirements-lock.txt`
   is a `pip freeze` snapshot of 54 installed distributions from a Python
   3.13.11 macOS arm64 environment, created on 2026-10-29. `pip freeze`
   reports installed versions; it does not solve requirements or create a
   cross-platform lockfile. `setup_env.sh` installs the snapshot. The venv,
   `.venv/`, is a build product and is not committed.
2. **Configuration.** `.env.example` lists every variable the scripts read
   (`DATA_PATH`, `RESULTS_PATH`, `OMP_NUM_THREADS`, `MPLBACKEND`, `LLM_BASE_URL`,
   `LLM_API_KEY`, `LLM_MODEL`, `ATTRACTOR_H`) with a default and a one-line comment.
   `setup_env.sh` copies it to `.env`, which is ignored by git and holds the
   local values; the scripts read `.env` with `python-dotenv`. A variable set
   in the shell takes precedence over `.env`, so
   `RESULTS_PATH=/tmp/out python analyse.py` writes elsewhere without editing
   the file.
3. **Setup script.** `setup_env.sh` uses these files to create a
   working shell: it creates `.venv` with `python3.13`, activates it, installs
   the package snapshot and then the project itself, editable and without
   dependencies (`pip install -e . --no-deps`), creates `.env` if missing, exports its variables
   (`set -a; . ./.env; set +a`), registers the Jupyter kernel
   `attractor-py313` (display name `Python (attractor 3.13)`, distinct from
   the `attractor` kernel of the first lectures) and prints `check_env.py`.

## Files

| File | Role |
|---|---|
| `data/C1.csv` ... `data/C10.csv` | the ten runs; columns: time (days), algae (10^6 cells/ml), rotifers (animals/ml), egg ratio, eggs, dead animals, medium |
| `pyproject.toml` | the package `attractor`: name, version, dependencies, the extras `fit` (SciPy) and `plot` (Matplotlib), the console commands, the package data |
| `src/attractor/__init__.py` | the public interface: the names listed in `__all__`, and `__version__` from the installed metadata |
| `src/attractor/cli.py` | the console commands `attractor-simulate`, `attractor-periods` (the table of periods of the runs, `results/periods.csv`; `--plot FILE` for the figure) and `attractor-fit` (the lynx–hare fit) |
| `src/attractor/datasets.py` | `load_run`, one chemostat run from its CSV file, and `lynx_hare`, the table in the package |
| `src/attractor/models.py` | abstract class `DynamicalSystem` and the models `LotkaVolterra`, `RosenzweigMacArthur` and `Goodwin`; the factory functions of Lecture 9, registered by name; the chemostat parameters `CHEMOSTAT` |
| `src/attractor/integrators.py` | Euler and RK4 steps, `integrate`, and the classes `Euler` and `RK4`, whose `run` returns a `Trajectory` |
| `src/attractor/cycles.py` | peak finder and cycle period |
| `simulate.py` | integrates the chemostat model with the loop written out, prints its period and writes `results/simulated.png`; step size from `ATTRACTOR_H`; `attractor-simulate` runs the same model |
| `puzzles/` | short scripts on names, calls, scopes, closures and imports (Lecture 9) |
| `exercises/lecture09/` | scripts that check the answers to the exercises of Lecture 9 |
| `src/attractor/tools.py` | decorators `counted`, `timed` and `register`, and `REGISTRY`, the model factories by name |
| `experiments/` | scripts of Lectures 10 and 11; with the project installed they also run as files from any folder |
| `exercises/lecture10/` | scripts that check the answers to the exercises of Lecture 10 |
| `src/attractor/trajectory.py` | class `Trajectory`: the times and states of one run |
| `src/attractor/data/lynx_hare.csv` | lynx and snowshoe hare pelts traded by the Hudson's Bay Company, 1900–1920, in thousands; package data read by `attractor.lynx_hare()` |
| `exercises/lecture11/` | scripts that check the answers to the exercises of Lecture 11 |
| `exercises/lecture12/` | scripts that check the answers to the exercises of Lecture 12 |
| `sessions/lecture12/` | interpreter sessions of the Lecture 12 notes, checked with `python -m doctest sessions/lecture12/*.txt` |
| `sessions/lecture11/` | interpreter sessions of the Lecture 11 notes, checked with `python -m doctest sessions/lecture11/*.txt` |
| `sessions/lecture10/` | interpreter sessions of the Lecture 10 notes, checked with `python -m doctest sessions/lecture10/*.txt` |
| `lecture10.ipynb` | the notebook of Lecture 10, run with the kernel `Python (attractor 3.13)` |
| `lecture10-mine.ipynb` | the notebook of Lecture 10 as run in the lecture, with its outputs |
| `lecture11.ipynb` | the notebook of Lecture 11, run with the same kernel |
| `lecture11-mine.ipynb` | the notebook of Lecture 11 as run in the lecture, with its outputs |
| `sessions/lecture09/` | interpreter sessions of the Lecture 9 notes, checked with `python -m doctest sessions/lecture09/*.txt` |
| `check_env.py` | interpreter, the versions and locations of NumPy and `attractor`, and the project variables |
| `ask_llm.py` | sends `results/periods.csv` to an OpenAI-compatible endpoint (`LLM_BASE_URL`) and reports the shortest and longest periods |
| `requirements.txt` | package specification: the five imported packages and ipykernel |
| `requirements-lock.txt` | `pip freeze` snapshot, Python 3.13.11 on macOS arm64 (Python 3.12.12 up to `lecture-09`) |
| `.env.example` | configuration template; copied to `.env` by `setup_env.sh` |
| `setup_env.sh` | one-step setup, to be sourced |
| `.gitignore` | `.venv*/`, `.env`, `results*/`, `__pycache__/`, `.ipynb_checkpoints/`, the build products `build/`, `dist/`, `*.egg-info/` (also `.venv313` and `results-np1`, `results-np2` of the lecture) |

## Python 3.12 and the NumPy 1.26.4 wheels

The course interpreter is Python 3.13. The scripts were written in 2024 against
NumPy 1.26, and the package snapshot pins `numpy==1.26.4`, the last 1.x release. NumPy
1.26.4 ships wheels for Python 3.9 to 3.12 only. On Python 3.13,
`pip install --only-binary=:all: -r requirements-lock.txt` stops with
`ERROR: No matching distribution found for numpy==1.26.4`; without
`--only-binary`, pip downloads `numpy-1.26.4.tar.gz` (15.8 MB) and compiles
it, which took 21 s on the lecture laptop and fails on a machine without a
C compiler.
With the specification `requirements.txt` instead, pip resolves `numpy>=1.26` to the
newest NumPy 2.x, and `analyse.py`, the command `attractor-periods` since Lecture 12, stops with

```
AttributeError: `np.NaN` was removed in the NumPy 2.0 release. Use `np.nan` instead.
```

Replacing `np.NaN` by `np.nan` makes the script run under NumPy 2. The table
on screen is unchanged, while the decimal strings in `results/periods.csv`
differ:
`C9,141,17,8.059374809265137` under NumPy 1.26.4 and `C9,141,17,8.059375`
under NumPy 2.5.4. The data are loaded as `float32`; the period is a
`float32` scalar divided by a Python `int`, which NumPy 1 promotes to
`float64` and NumPy 2 keeps in `float32` (NEP 50). For C9, both strings
convert to the same binary value, 8.059374809265137; NumPy 2 writes the
shortest decimal that round-trips to that value. The 2024 output is
reproduced with Python 3.12 and the package snapshot.

From `lecture-10` the project runs on Python 3.13 with NumPy 2, from a new
snapshot of `requirements.txt`. The 2024 snapshot stays at the tags up to
`lecture-09`.

## Method

The algae density of each run is read with `numpy.genfromtxt`; samples
recorded as `NaN` are dropped. A peak is a local maximum above the run's mean
density, at least three days after the previous peak. The period is the time
between the first and the last peak divided by the number of intervals. For
C9 (141 samples, 17 peaks) this gives 8.059 days, the period of the nutrient
pulses in that run.

## Data

The ten runs C1 to C10 are the chemostat experiments of B. Blasius, L. Rudolf,
G. Weithoff, U. Gaedke and G. F. Fussmann, *Long-term cyclic persistence in an
experimental predator-prey system*, Nature **577**, 226-230 (2020),
<https://www.nature.com/articles/s41586-019-1857-0>; data set on figshare,
doi:10.6084/m9.figshare.10045976, <https://doi.org/10.6084/m9.figshare.10045976>
(CC BY 4.0).

`src/attractor/data/lynx_hare.csv` is the data file of B. Carpenter, *Predator-Prey
Population Dynamics: the Lotka-Volterra model in Stan* (2018), from
<https://github.com/stan-dev/example-models>
(`knitr/lotka-volterra/hudson-bay-lynx-hare.csv`, BSD licence), which takes
the table from P. Howard's lecture notes (2009).

## Licence

MIT, Boris Bolliet 2026 (see `LICENSE`); the chemostat data keep their CC BY 4.0 licence and the lynx–hare table its BSD licence.
