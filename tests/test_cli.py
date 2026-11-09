"""Integration tests: the console commands and the data installed with the package."""
import re

import pytest

from attractor import lynx_hare
from attractor.cli import fit, simulate


def test_simulate(capsys):
    simulate(["--h", "0.05", "--days", "40"])
    out = capsys.readouterr().out
    m = re.match(r"801 points, step 0.05, algae period (\S+) days", out)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(8.2, abs=0.1)


def test_lynx_hare_is_installed():
    year, lynx, hare = lynx_hare()
    assert year[0] == 1900
    assert len(year) == len(lynx) == len(hare) == 21


def test_fit(capsys):
    pytest.importorskip("scipy")
    fit([])
    assert "rms residual" in capsys.readouterr().out
