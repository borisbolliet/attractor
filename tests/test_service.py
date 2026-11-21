"""The explorer's API and its record of runs, with SQLite in a temporary file."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("scipy")
from fastapi.testclient import TestClient

from attractor.service import app as service
from attractor.service.store import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'runs.db'}")
    service.store.cache_clear()  # a new Store for this test's database
    yield TestClient(service.app)
    service.store.cache_clear()


def test_page_and_assets_come_from_the_package(client):
    assert client.get("/oregonator/").text.startswith("<!doctype html>")
    assert client.get("/oregonator/assets/explorer.js").status_code == 200


def test_simulate_returns_and_records_the_trajectory(client):
    reply = client.post("/oregonator/simulate/oregonator", json={"t_end": 30})
    assert reply.status_code == 200
    path = reply.json()
    assert path["t"][0] == 0 and path["t"][-1] == pytest.approx(30)
    [run] = client.get("/oregonator/runs").json()
    assert (run["t_end"], run["f"], run["steps"]) == (30, 1, len(path["t"]) - 1)


@pytest.mark.parametrize("path, body, status", [
    ("/oregonator/simulate/Oregonator", {"t_end": 30}, 404),
    ("/oregonator/simulate/Oregonator", {"t_end": 0}, 422),
    ("/oregonator/simulate/oregonator", {"t_end": 30, "steps": 5}, 422),
])
def test_status_codes(client, path, body, status):
    assert client.post(path, json=body).status_code == status


def test_latest_runs_come_newest_first(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'runs.db'}")
    for t_end in (10, 20, 30):
        store.add(t_end, 1.0, 100, 0.01)
    assert [run["t_end"] for run in store.latest(2)] == [30, 20]


def test_unknown_database_is_refused():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Store("mysql://localhost/runs")
