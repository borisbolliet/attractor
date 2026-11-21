"""A record of the simulations the explorer has run, in a database.

DATABASE_URL chooses the database: ``sqlite:///runs.db`` (the default) is a
file opened by Python's sqlite3 module; ``postgresql://user:password@host/name``
is a PostgreSQL server, reached with psycopg (the ``postgres`` extra).
"""
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

COLUMNS = ("started", "t_end", "f", "steps", "seconds")
# The columns are the same in both databases; only the generated id differs.
TABLE = """CREATE TABLE IF NOT EXISTS runs (
    id {id},
    started TEXT NOT NULL,
    t_end DOUBLE PRECISION NOT NULL,
    f DOUBLE PRECISION NOT NULL,
    steps INTEGER NOT NULL,
    seconds DOUBLE PRECISION NOT NULL
)"""
ID = {"sqlite": "INTEGER PRIMARY KEY",
      "postgresql": "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY"}
MARK = {"sqlite": "?", "postgresql": "%s"}  # each driver's parameter placeholder


class Store:
    def __init__(self, url=None):
        self.url = url or os.environ.get("DATABASE_URL", "sqlite:///runs.db")
        if self.url.startswith("sqlite:///"):
            self.kind = "sqlite"
        elif self.url.startswith("postgresql://"):
            self.kind = "postgresql"
        else:
            raise ValueError("DATABASE_URL must start with sqlite:/// or postgresql://")
        with closing(self.connect()) as db, db:
            db.execute(TABLE.format(id=ID[self.kind]))

    def connect(self):
        if self.kind == "sqlite":
            return sqlite3.connect(self.url.removeprefix("sqlite:///"))
        import psycopg  # installed with the postgres extra

        return psycopg.connect(self.url)

    def add(self, t_end, f, steps, seconds):
        """Insert one run; `with db` commits it, `closing` closes the connection."""
        started = datetime.now(UTC).isoformat(timespec="seconds")
        marks = ", ".join([MARK[self.kind]] * len(COLUMNS))
        with closing(self.connect()) as db, db:
            db.execute(f"INSERT INTO runs ({', '.join(COLUMNS)}) VALUES ({marks})",
                       (started, t_end, f, steps, seconds))

    def latest(self, n=5):
        """The n most recent runs, newest first, as dictionaries."""
        query = f"SELECT id, {', '.join(COLUMNS)} FROM runs ORDER BY id DESC LIMIT {MARK[self.kind]}"
        with closing(self.connect()) as db:
            rows = db.execute(query, (n,)).fetchall()
        return [dict(zip(("id",) + COLUMNS, row)) for row in rows]
