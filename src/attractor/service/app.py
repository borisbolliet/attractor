"""A FastAPI application that runs the Oregonator and returns its trajectory.

The page, its browser files and the API are all served under /oregonator,
as on the course site (Lecture 20). Each simulation is recorded in the
database named by DATABASE_URL (store.py).
"""
from functools import cache
from importlib.resources import files
from time import perf_counter
from typing import Annotated, TypedDict

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi import Path as PathParameter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from attractor import Adaptive, Oregonator

from .store import Store

PREFIX = "/oregonator"
STATIC = files("attractor.service") / "static"  # package data, found wherever the package is installed

app = FastAPI(title="Oregonator API", docs_url=f"{PREFIX}/docs",
              openapi_url=f"{PREFIX}/openapi.json", redoc_url=None)
router = APIRouter(prefix=PREFIX)


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    t_end: float = Field(default=30, ge=1, le=360)
    f: float = Field(default=1, ge=0.8, le=1.2)


class Trajectory(TypedDict):
    t: list[float]
    alpha: list[float]
    eta: list[float]
    rho: list[float]


@cache
def store() -> Store:
    """One Store per process, created at the first request that needs it."""
    return Store()


def trajectory(run: Run, method: str = "BDF") -> Trajectory:
    path = Adaptive(method, rtol=1e-6, atol=1e-6).run(
        Oregonator(f=run.f), [1.0, 2.0, 3.0], 0.0, run.t_end)
    return {"t": path.t.tolist(), "alpha": path.y[:, 0].tolist(),
            "eta": path.y[:, 1].tolist(), "rho": path.y[:, 2].tolist()}


@router.post("/simulate/{model}")
def simulate(model: Annotated[str, PathParameter(examples=["oregonator"])], run: Run) -> Trajectory:
    """Use `oregonator` for the model. Returns model times (`t`) and the
    three concentrations (`alpha`, `eta`, `rho`) at those times."""
    if model != "oregonator":
        raise HTTPException(404, "Unknown model")
    start = perf_counter()
    path = trajectory(run)
    store().add(run.t_end, run.f, len(path["t"]) - 1, perf_counter() - start)
    return path


@router.get("/runs")
def runs(limit: Annotated[int, Query(ge=1, le=100)] = 5) -> list[dict]:
    """The most recent simulations, newest first."""
    return store().latest(limit)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/", include_in_schema=False)
def explorer():
    return HTMLResponse(STATIC.joinpath("index.html").read_text(encoding="utf-8"))


@app.get(PREFIX, include_in_schema=False)
def canonical_path():
    return RedirectResponse(PREFIX + "/", status_code=308)


app.include_router(router)
app.mount(PREFIX + "/assets", StaticFiles(packages=[("attractor.service", "static")]), name="assets")
