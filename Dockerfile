FROM python:3.13-slim
WORKDIR /srv
COPY . .
RUN python -m pip install --no-cache-dir ".[service]"
EXPOSE 8000
CMD ["attractor-serve"]
