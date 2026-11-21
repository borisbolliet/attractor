"""The console command attractor-serve: Uvicorn running the explorer."""
import os


def main():
    import uvicorn  # part of the service extra

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("attractor.service.app:app", host=host, port=port)
