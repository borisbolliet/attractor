"""pytest collects the doctests of src/ (--doctest-modules), importing every
module. The service modules import FastAPI, which only the service extra
installs, so without it they are left out of the collection."""
import importlib.util

collect_ignore = [] if importlib.util.find_spec("fastapi") else ["src/attractor/service"]
