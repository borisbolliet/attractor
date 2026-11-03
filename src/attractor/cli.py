"""Console commands of the package, declared in [project.scripts] of pyproject.toml."""
import argparse
import csv
import importlib
import os
from pathlib import Path

import numpy as np

from .cycles import cycle_period
from .datasets import load_run, lynx_hare
from .integrators import RK4
from .models import LotkaVolterra


def _optional(module, extra):
    """Import an optional dependency, or stop with the command that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        name = module.split(".")[0]
        hint = f'pip install "attractor[{extra}]"'
        raise SystemExit(f"{name} is not installed: {hint}") from None


def simulate(argv=None):
    """Run the chemostat model with RK4 and print the period of the algae."""
    parser = argparse.ArgumentParser(prog="attractor-simulate",
                                     description=simulate.__doc__)
    parser.add_argument("--h", type=float,
                        default=float(os.environ.get("ATTRACTOR_H", "0.05")),
                        help="step size in days (default: $ATTRACTOR_H, or 0.05)")
    parser.add_argument("--days", type=float, default=40.0,
                        help="length of the run in days (default: 40)")
    args = parser.parse_args(argv)
    tr = RK4(args.h).run(LotkaVolterra(), (1.5, 8.0), 0.0, args.days)
    period = tr.period()
    print(f"{len(tr)} points, step {args.h:g}, algae period {period:.2f} days")


def periods(argv=None):
    """Print the cycle period of chemostat runs, from the peaks of the algae density."""
    parser = argparse.ArgumentParser(prog="attractor-periods",
                                     description=periods.__doc__)
    parser.add_argument("runs", nargs="*", type=Path,
                        help="CSV files of runs (default: every C*.csv in $DATA_PATH, or data/)")
    parser.add_argument("--plot", type=Path, metavar="FILE",
                        help="also draw the peak times into FILE (needs the plot extra)")
    args = parser.parse_args(argv)
    data = Path(os.environ.get("DATA_PATH", "data"))
    runs = args.runs or sorted(data.glob("C*.csv"), key=lambda f: int(f.stem[1:]))
    if not runs:
        raise SystemExit(f"attractor-periods: no run files given and no C*.csv in {data}")
    rows, peak_times = [], []
    print(f"{'run':<5}{'samples':>9}{'peaks':>7}{'period (days)':>15}")
    for path in runs:
        t, algae = load_run(path)
        period, tp = cycle_period(t, algae)
        rows.append((path.stem, len(t), len(tp), period))
        peak_times.append(tp)
        print(f"{path.stem:<5}{len(t):>9}{len(tp):>7}{period:>15.3f}")
    results = Path(os.environ.get("RESULTS_PATH", "results"))
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "periods.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("run", "samples", "peaks", "period_days"), *rows])
    print(f"written: {results / 'periods.csv'}")
    if args.plot:
        _plot_peaks(rows, peak_times, args.plot)


def _plot_peaks(rows, peak_times, path):
    """Draw the peak times of each run, one row per run, into the file path."""
    matplotlib = _optional("matplotlib", "plot")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, tp in enumerate(peak_times):
        ax.plot(tp, np.full(len(tp), i), "|", ms=10, color="C0")
    ax.set_yticks(range(len(rows)), [f"{r[0]}  ({r[3]:.2f} d)" for r in rows])
    ax.set_xlabel("time (days)")
    ax.set_title("Algae peaks of each run")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"written: {path}")


P0 = (0.5, 0.02, 0.01, 0.8, 30.0, 4.0)  # a, b, c, d, and the hare and lynx pelts of 1900


def fit(argv=None):
    """Fit the Lotka-Volterra model to the lynx and hare pelts in the package."""
    argparse.ArgumentParser(prog="attractor-fit",
                            description=fit.__doc__).parse_args(argv)
    least_squares = _optional("scipy.optimize", "fit").least_squares
    year, lynx, hare = lynx_hare()

    def residuals(p):  # p = (a, b, c, d, H0, L0); h = 0.02 gives 50 steps per year
        tr = RK4(0.02).run(LotkaVolterra(*p[:4]), p[4:], 0.0, 20.0)
        return np.concatenate([tr.y[::50, 0] - hare, tr.y[::50, 1] - lynx])

    result = least_squares(residuals, P0, bounds=(0, np.inf))
    a, b, c, d, h0, l0 = result.x
    rms = np.sqrt(np.mean(result.fun**2))
    print(f"LotkaVolterra(a={a:.3f}, b={b:.4f}, c={c:.4f}, d={d:.3f})")
    print(f"H0 = {h0:.2f}, L0 = {l0:.2f}, rms residual {rms:.2f} thousand pelts")
