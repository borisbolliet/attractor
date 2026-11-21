FROM python:3.13-slim
WORKDIR /srv
RUN mkdir /data

# Dependencies first: this layer is rebuilt only when requirements-service.txt changes.
COPY requirements-service.txt .
RUN python -m pip install --no-cache-dir -r requirements-service.txt

# The package itself, without resolving its dependencies again.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir --no-deps .

# Listen on every interface of the container; keep the SQLite file in /data.
ENV HOST=0.0.0.0 PORT=8000 DATABASE_URL=sqlite:////data/runs.db
EXPOSE 8000
CMD ["attractor-serve"]
